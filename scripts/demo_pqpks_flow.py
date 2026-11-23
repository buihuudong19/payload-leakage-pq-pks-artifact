"""Demonstration of the PQ-PKS flow as designed (paper, Section 6, Figs. 4-6) on synthetic test covers (pqpks.data).

Scenario: one session of N = 3 images.
  Image 1  (test cover T1): session start, carries c_hat (stage 1) + t0, a text, an ML-DSA-65 signature (stage 2)
  Image 2  (test cover T2): a binary message of 919 bytes
  Image 3  (test cover T3): a binary message of 2420 bytes, end-of-session flag
The receiver requires authentication (require_auth) and receives out of order (1, 3, 2), then runs negative cases: a plain
image, a wrong key, replays, one modified pixel in A0.
Output: results/archive/pre-confirm/demo/ with the stego images (PNG), report.json, REPORT.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pqpks import data, systems  # noqa: E402
from pqpks.analysis import dpub, metrics  # noqa: E402
from pqpks.params import DEFAULT_PP, FRAME_HEADER_BYTES, KEMELEON_CT_BYTES, START_HEADER_BYTES  # noqa: E402
from pqpks.stego import partition  # noqa: E402

OUT = ROOT / "results" / "archive" / "pre-confirm" / "demo"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = data.cover_pool()
    sysm = systems.make("PQ-PKS")
    log: dict = {"profile_id": sysm.pid.hex(), "sig_alg": DEFAULT_PP.sig_alg, "images": [], "receiver": [],
                 "negative": []}

    t = time.perf_counter()
    dk, ek = sysm.keygen()
    sk_s, vk_s = sysm.sig_keygen()
    log["keygen_ms"] = round(1000 * (time.perf_counter() - t), 2)
    log["sizes"] = {"ek_R": len(ek), "vk_S": len(vk_s)}

    t = time.perf_counter()
    sess = sysm.start_session(ek, sk_s=sk_s, length=3)
    log["start_session_ms"] = round(1000 * (time.perf_counter() - t), 2)
    log["session"] = {"kem_part_bytes": len(sess.kem_part), "sid": sess.sid.hex()[:16] + "…",
                      "kem_part_uniformity": dpub.generic_uniformity(sess.kem_part),
                      "kem_part_mlkem_llr": dpub.mlkem_raw_llr(sess.kem_part)}

    plan = [("T1", b"PQ-PKS: text message in the session-start image.", "text + ML-DSA-65 signature"),
            ("T2", data.test_messages()[0], "binary message, 919 B"),
            ("T3", data.test_messages()[1], "binary message, 2420 B, end of session")]
    stegos = []
    for i, (cname, msg, label) in enumerate(plan, start=1):
        cover = pool[cname]
        t = time.perf_counter()
        stego = sysm.send(sess, msg, cover)
        dt = time.perf_counter() - t
        stegos.append((stego, msg))
        Image.fromarray(stego).save(OUT / f"stego_{i}_{cname}.png")
        n = cover.size
        body_bytes = None
        for b in DEFAULT_PP.buckets:
            head = START_HEADER_BYTES + DEFAULT_PP.sig_bytes if i == 1 else FRAME_HEADER_BYTES
            if head + len(msg) <= b:
                body_bytes = b + 16
                break
        kem_bits = 8 * KEMELEON_CT_BYTES if i == 1 else 0
        n0 = partition.stage_sizes(n, kem_bits, DEFAULT_PP.alpha0) if i == 1 else 0
        entry = {"index": i, "cover": cname, "message": label, "message_bytes": len(msg),
                 "stage1_bits": kem_bits, "stage1_pixels": n0,
                 "stage1_alpha": round(kem_bits / n0, 4) if n0 else None,
                 "stage2_bits": 8 * body_bytes, "stage2_pixels": n - n0,
                 "stage2_alpha": round(8 * body_bytes / (n - n0), 4),
                 "total_bpp": round((kem_bits + 8 * body_bytes) / n, 4),
                 **metrics.distortion(cover, stego), "send_s": round(dt, 2)}
        log["images"].append(entry)

    st = systems.ReceiverState()
    pol = systems.ReceiverPolicy(require_auth=True, vk_s=vk_s)
    for i in (1, 3, 2):
        stego, msg = stegos[i - 1]
        t = time.perf_counter()
        r = sysm.receive(dk, stego, st, pol, ek)
        log["receiver"].append({"image": i, "status": r.status, "index": r.index, "authenticated": r.authenticated,
                                "message_ok": r.msg == msg, "recv_s": round(time.perf_counter() - t, 2)})

    # --- negative cases
    neg = log["negative"]
    neg.append({"case": "plain image (cover T1)",
                "status": sysm.receive(dk, pool["T1"], systems.ReceiverState()).status})
    dk_other, _ = sysm.keygen()
    neg.append({"case": "another receiver (wrong dk)",
                "status": sysm.receive(dk_other, stegos[0][0], systems.ReceiverState()).status})
    neg.append({"case": "replay of image 3", "status": sysm.receive(dk, stegos[2][0], st, pol, ek).status})
    neg.append({"case": "replay of image 1", "status": sysm.receive(dk, stegos[0][0], st, pol, ek).status})
    tam = stegos[0][0].copy().reshape(-1)
    a0, _ = partition.partition(512, 512, DEFAULT_PP.context, 8 * KEMELEON_CT_BYTES, DEFAULT_PP.alpha0)
    p = int(a0[0])
    tam[p] = tam[p] + 1 if tam[p] < 255 else tam[p] - 1
    neg.append({"case": "1 pixel modified in A0 of image 1",
                "status": sysm.receive(dk, tam.reshape(512, 512), systems.ReceiverState()).status})

    # --- public extraction: the same extraction on stego image 1 and on the cover (what a warden can do)
    emb = sysm.embedder
    pub_stego = emb.extract_kem(stegos[0][0], KEMELEON_CT_BYTES)
    pub_cover = emb.extract_kem(pool["T1"], KEMELEON_CT_BYTES)
    log["public_extraction"] = {
        "stego_equals_kem_part": pub_stego == sess.kem_part,
        "stego": dpub.generic_uniformity(pub_stego), "cover": dpub.generic_uniformity(pub_cover),
        "stego_llr": dpub.mlkem_raw_llr(pub_stego), "cover_llr": dpub.mlkem_raw_llr(pub_cover)}

    (OUT / "report.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    (OUT / "REPORT.md").write_text(render(log), encoding="utf-8")
    print(render(log))


def render(log) -> str:
    L = ["# Demonstration of the PQ-PKS flow", "",
         f"Profile: signature {log['sig_alg']}; profile_id = `{log['profile_id'][:16]}…`; the receiver requires authentication (require_auth)", "",
         f"- Key generation (ML-KEM-768 + {log['sig_alg']}): {log['keygen_ms']} ms; |ek_R| = {log['sizes']['ek_R']} B, |vk_S| = {log['sizes']['vk_S']} B",
         f"- Session start (HPKE.SetupBaseS + Kemeleon + transcript signature): {log['start_session_ms']} ms; |ĉ| = {log['session']['kem_part_bytes']} B",
         f"- Uniformity of ĉ: {log['session']['kem_part_uniformity']}; LLR(raw ML-KEM) = {log['session']['kem_part_mlkem_llr']:.2f}", "",
         "## Sender", "",
         "| Image | Cover | Message | Stage 1 bits / pixels / α₀ | Stage 2 bits / pixels / α | Total bpp | Changed pixels | Change rate | PSNR (dB) | max|Δ| | Time (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in log["images"]:
        L.append(f"| {e['index']} | {e['cover']} | {e['message']} ({e['message_bytes']} B) | "
                 f"{e['stage1_bits']} / {e['stage1_pixels']} / {e['stage1_alpha']} | "
                 f"{e['stage2_bits']} / {e['stage2_pixels']} / {e['stage2_alpha']} | {e['total_bpp']} | "
                 f"{e['changed']} | {e['change_rate']:.4f} | {e['psnr']:.2f} | {e['max_abs_diff']} | {e['send_s']} |")
    L += ["", "## Receiver (arrival order: 1, 3, 2)", "", "| Image | Status | Index | Authenticated | Message correct | Time (s) |", "|---|---|---|---|---|---|"]
    for r in log["receiver"]:
        L.append(f"| {r['image']} | {r['status']} | {r['index']} | {r['authenticated']} | {r['message_ok']} | {r['recv_s']} |")
    L += ["", "## Negative cases", "", "| Case | Result |", "|---|---|"]
    for n in log["negative"]:
        L.append(f"| {n['case']} | {n['status']} |")
    pe = log["public_extraction"]
    L += ["", "## Public extraction H₀·LSB(y[A₀]) (what a warden can do)", "",
          f"- On stego image 1 the extracted string equals ĉ: {pe['stego_equals_kem_part']}",
          f"- Stego image: {pe['stego']}; LLR = {pe['stego_llr']:.2f}",
          f"- Cover image: {pe['cover']}; LLR = {pe['cover_llr']:.2f}",
          "", "_A single sample does not support a statistical conclusion; the full tests belong to E1, E2._"]
    return "\n".join(L)


if __name__ == "__main__":
    main()
