"""SRNet training for gate G4 and for the confirmatory step. Runs in a separate GPU environment (.venv-gpu: torch, numpy).

All parameters of each training run are in configs/srnet_runs_v2.json (registered in advance, frozen with the profile):
  - Adamax, 16 cover-stego pairs per batch, L2 2e-4 on the convolution weights, BN with running-average factor 0.9 (PyTorch
    momentum 0.1), the same random rotation by k*90 degrees and flip for both images of a pair; schedule of [BCF19] Section III-D:
    0.4 bpp from scratch, 400k iterations at lr 1e-3 + 100k iterations at lr 1e-4; lower rates: curriculum from the 0.4 bpp model.
  - the snapshot with the highest validation accuracy within the last `select_last` iterations is kept (validation every epoch).
  - confirmatory step: training on random 256x256 crops (the same position for the cover and the stego of a pair) of the
    512x512 images, evaluation on the full 512x512 images; PQ-PKS and REF-U use the same images (the intersection of the two
    stego sets) and the same seed, hence the same batch order and the same augmentation.
Score of an image: logit(stego) - logit(cover). Output of a run in results/srnet/<name>/:
  state_last.pt (for resuming), best.pt, log.jsonl, scores.npz (validation and test scores of the selected snapshot), done.json.
After an interruption (Ctrl+C, power loss): rerun the same command; the script reads state_last.pt and resumes from the saved iteration.

SRNet G4 (HILL 0.4 bpp only, short branch first):
    srnet_train.py --run g4_hill_0.4 --until 200000     # 200k iterations at lr 1e-3, saves state_200000.pt and pauses
    srnet_train.py --run g4_hill_0.4_short              # continues from iteration 200000: 50k iterations at lr 1e-4
    srnet_evaluate.py g4                                 # pass: use the short branch; fail: run the next command
    srnet_train.py --run g4_hill_0.4                    # continues from iteration 200000 to the end of the [BCF19] schedule (500k)
    srnet_train.py --sequence confirm                   # after SRNet G4 has passed
    srnet_train.py --plan --its 5                       # schedule and estimated time
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pqpks.steganalysis.srnet import SRNet  # noqa: E402

CFG_P = ROOT / "configs" / "srnet_runs_v2.json"
FROZEN = ROOT / "configs" / "profile_frozen_v1.json"
CHECKED = ("configs/srnet_runs_v2.json", "configs/g4_srnet_reference.json", "src/pqpks/steganalysis/srnet.py")
EQUIV = ROOT / "provenance" / "release_equivalence_v1.json"


def check_frozen() -> None:
    """Refuse to run if the SRNet configuration or the model code differs from the frozen profile (no full pqpks package needed).
    A released file is accepted when provenance/release_equivalence_v1.json links its hash to the frozen one."""
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["source_sha256"]
    equiv = json.loads(EQUIV.read_text(encoding="utf-8"))["files"] if EQUIV.exists() else {}

    def ok(k):
        h = hashlib.sha256((ROOT / k).read_bytes()).hexdigest()
        e = equiv.get(k)
        return h == frozen.get(k) or (e is not None and e["frozen_sha256"] == frozen.get(k) and h == e["release_sha256"])
    bad = [k for k in CHECKED if not ok(k)]
    if bad:
        raise SystemExit(f"differs from the frozen profile profile_frozen_v1: {bad}")


# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------
class PairData:
    def __init__(self, cfg: dict, run: dict):
        ds = cfg["datasets"][run["data"]]
        d = ROOT / ds["dir"]
        self.index = json.loads((d / "index.json").read_text(encoding="utf-8"))
        self.cover = np.load(d / "cover.npy", mmap_mode="r")
        self.stego = np.load(d / f"{run['stego']}.npy", mmap_mode="r")
        self.size = ds["size"]
        avail = None
        if "available" in self.index:                       # confirmatory step: only images with a stego (of both systems when paired)
            avail = set(self.index["available"][run["stego"]])
            if run.get("pair_with"):
                avail &= set(self.index["available"][run["pair_with"]])
        pick = (lambda lst: np.array([p for p in lst if avail is None or p in avail], dtype=np.int64))
        self.train, self.val, self.test = pick(self.index["train"]), pick(self.index["val"]), pick(self.index["test"])
        self.ids = self.index["ids"]


class Batches:
    """Batch t depends only on (seed, t): after an interruption the batch order and the augmentation are the same as without it."""

    def __init__(self, data: PairData, pairs: int, seed: int, crop: int | None):
        self.d, self.B, self.seed, self.crop = data, pairs, seed, crop
        self.per_epoch = len(data.train) // pairs
        self._perm = (None, None)

    def get(self, t: int):
        ep, j = divmod(t, self.per_epoch)
        if self._perm[0] != ep:
            self._perm = (ep, np.random.default_rng([self.seed, 7, ep]).permutation(len(self.d.train)))
        sel = self.d.train[self._perm[1][j * self.B:(j + 1) * self.B]]
        order = np.argsort(sel)                             # read the memmap in increasing order
        c = np.empty((self.B,) + self.d.cover.shape[1:], np.uint8)
        s = np.empty_like(c)
        c[order], s[order] = self.d.cover[sel[order]], self.d.stego[sel[order]]
        rng = np.random.default_rng([self.seed, 11, t])
        S = c.shape[-1]
        k, fl = rng.integers(0, 4, self.B), rng.integers(0, 2, self.B)
        if self.crop:
            r, q = rng.integers(0, S - self.crop + 1, (2, self.B))
            c = np.stack([c[b, r[b]:r[b] + self.crop, q[b]:q[b] + self.crop] for b in range(self.B)])
            s = np.stack([s[b, r[b]:r[b] + self.crop, q[b]:q[b] + self.crop] for b in range(self.B)])
        x = np.empty((2 * self.B,) + c.shape[1:], np.uint8)
        for b in range(self.B):
            cc, ss = np.rot90(c[b], k[b]), np.rot90(s[b], k[b])
            if fl[b]:
                cc, ss = cc[:, ::-1], ss[:, ::-1]
            x[2 * b], x[2 * b + 1] = cc, ss
        y = np.tile(np.array([0, 1], np.int64), self.B)
        return x, y


class Prefetch:
    def __init__(self, batches: Batches, start: int, stop: int, depth: int = 6):
        self.q: queue.Queue = queue.Queue(depth)
        self._t = threading.Thread(target=self._run, args=(batches, start, stop), daemon=True)
        self._t.start()

    def _run(self, batches, start, stop):
        for t in range(start, stop):
            self.q.put(batches.get(t))

    def get(self):
        return self.q.get()


# ----------------------------------------------------------------------------------------------------------------------
# Training and scoring
# ----------------------------------------------------------------------------------------------------------------------
def to_dev(x: np.ndarray, dev) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True).float().unsqueeze(1)
    return t.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def score(model, arr, positions, dev, amp: bool, bs: int) -> np.ndarray:
    model.eval()
    out = []
    for k in range(0, len(positions), bs):
        p = np.sort(positions[k:k + bs])
        inv = np.argsort(np.argsort(positions[k:k + bs]))
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            z = model(to_dev(arr[p], dev)).float()
        out.append((z[:, 1] - z[:, 0]).cpu().numpy()[inv])
    model.train()
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def val_metrics(model, data: PairData, dev, amp, bs):
    sc = score(model, data.cover, data.val, dev, amp, bs)
    ss = score(model, data.stego, data.val, dev, amp, bs)
    acc = 0.5 * ((sc <= 0).mean() + (ss > 0).mean())
    return float(acc), sc, ss


def lr_at(phases, t: int) -> float:
    for n, lr in phases:
        if t < n:
            return lr
        t -= n
    return phases[-1][1]


def param_groups(model: nn.Module, wd: float):
    conv_w = [m.weight for m in model.modules() if isinstance(m, nn.Conv2d)]
    ids = {id(p) for p in conv_w}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return [{"params": conv_w, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}]


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def smoke_iters(k: int) -> int:
    """Code test: scale every iteration count by the same factor so that the short branch still matches the stop point of the full run."""
    return max(2, k // 25000)


def resolve_init(init: str | None, out_root: Path) -> str | None:
    """'@g4_selected': the G4 run selected by the registered rule (selected_run in g4_srnet_summary.json)."""
    if init != "@g4_selected":
        return init
    s = out_root / "g4_srnet_summary.json"
    info = json.loads(s.read_text(encoding="utf-8")) if s.exists() else {}
    if not info.get("G4_SRNet_pass") or not info.get("selected_run"):
        raise SystemExit("SRNet G4 has not passed or no run is selected (results/srnet/g4_srnet_summary.json)")
    return info["selected_run"]


def run_one(cfg: dict, name: str, dev, out_root: Path, smoke: bool, until: int | None = None) -> None:
    run = dict(cfg["runs"][name])
    com = cfg["common"]
    out = out_root / name
    if (out / "done.json").exists():
        print(f"[{name}] already finished, skipped", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    phases = [tuple(p) for p in run["phases"]]
    val_every, monitor_every, select_last = com["val_every"], com["monitor_every"], run["select_last"]
    init_state = dict(run["init_state"]) if run.get("init_state") else None
    if smoke:                                               # code test only: very short schedule, same proportions
        phases = [(smoke_iters(n), lr) for n, lr in phases]
        val_every, monitor_every, select_last = 2, 4, smoke_iters(select_last)
        until = None if until is None else smoke_iters(until)
        if init_state:
            init_state["iter"] = smoke_iters(init_state["iter"])
    total = sum(n for n, _ in phases)
    if until is not None and not 0 < until < total:
        raise SystemExit(f"[{name}] --until must lie in (0, {total})")
    amp = bool(com["amp"]) and dev.type == "cuda"
    data = PairData(cfg, run)
    pairs = 2 if smoke else com["batch_pairs"]          # code test on CPU: small batch to fit in RAM
    batches = Batches(data, pairs, com["seed"], run.get("train_crop"))
    if batches.per_epoch < 1:
        raise SystemExit(f"[{name}] training set too small ({len(data.train)} images)")
    torch.manual_seed(com["seed"])
    model = SRNet().to(dev).to(memory_format=torch.channels_last)
    init = resolve_init(run["init"], out_root)
    if init:
        init_p = out_root / init / "best.pt"
        if not (out_root / init / "done.json").exists():
            raise SystemExit(f"[{name}] {init} must be finished first (initialization from that model)")
        model.load_state_dict(torch.load(init_p, map_location=dev, weights_only=False)["model"])
    opt = torch.optim.Adamax(param_groups(model, com["weight_decay_conv"]), lr=phases[0][1],
                             betas=tuple(com["optimizer"]["betas"]), eps=com["optimizer"]["eps"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    t0, best = 0, {"acc": -1.0, "iter": None}
    last_p, best_p = out / "state_last.pt", out / "best.pt"
    if last_p.exists():
        st = torch.load(last_p, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        t0, best = st["iter"], st["best"]
        print(f"[{name}] resuming from iteration {t0}", flush=True)
    elif init_state:                                         # short branch: continue from the state of the full run
        src = out_root / init_state["run"] / f"state_{init_state['iter']}.pt"
        if not src.exists():
            raise SystemExit(f"[{name}] {src} not found: first run srnet_train.py --run {init_state['run']} "
                             f"--until {init_state['iter']}")
        if phases[0][0] != init_state["iter"]:
            raise SystemExit(f"[{name}] the first phase ({phases[0][0]} iterations) must equal the branch point {init_state['iter']}")
        st = torch.load(src, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        scaler.load_state_dict(st["scaler"])
        t0 = st["iter"]
        print(f"[{name}] branching from {src.parent.name} at iteration {t0}", flush=True)
    stop_at = total if until is None else until
    if t0 >= stop_at:
        print(f"[{name}] already at iteration {t0} (stop at {stop_at}), skipped", flush=True)
        return
    bs_eval = 4 if smoke else (64 if data.size <= 256 else 16)
    print(f"[{name}] {run['stego']}: train {len(data.train)}, val {len(data.val)}, test {len(data.test)} images; "
          f"{total} iterations (this run up to {stop_at}), {batches.per_epoch} iterations per epoch; initialization {init or 'random'}; "
          f"training crop {run.get('train_crop') or data.size}; AMP {amp}", flush=True)
    log = open(out / "log.jsonl", "a", encoding="utf-8")
    pf = Prefetch(batches, t0, stop_at)
    model.train()
    tick, run_loss, n_loss = time.time(), torch.zeros((), device=dev), 0
    completed = t0                                           # number of iterations whose weight update is complete

    def save_state(t_done):
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                     "iter": t_done, "best": best}, last_p)

    stop = {"flag": False}

    def on_sigint(signum, frame):                            # Ctrl+C: only set a flag; stop at the start of the next iteration
        stop["flag"] = True

    old = signal.signal(signal.SIGINT, on_sigint)
    try:
        for t in range(t0, stop_at):
            if stop["flag"]:
                raise KeyboardInterrupt
            lr = lr_at(phases, t)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = pf.get()
            xb, yb = to_dev(x, dev), torch.from_numpy(y).to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logits = model(xb)
            loss = F.cross_entropy(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            completed = t + 1
            run_loss, n_loss = run_loss + loss.detach(), n_loss + 1          # summed on the GPU, no synchronization per iteration
            done = t + 1
            if done % (100 if not smoke else 5) == 0 or done == stop_at:
                el = time.time() - tick
                rate = n_loss / el if el > 0 else float("nan")
                eta = (stop_at - done) / rate / 3600 if rate > 0 else float("nan")      # up to the stop point of this run
                print(f"[{name}] iteration {done}/{stop_at} loss {float(run_loss) / n_loss:.4f} lr {lr:g} {rate:.2f} it/s "
                      f"about {eta:.1f} h left", flush=True)
                tick, run_loss, n_loss = time.time(), torch.zeros((), device=dev), 0
            in_window = done > total - select_last
            if (in_window and done % val_every == 0) or done % monitor_every == 0 or done == total:
                acc, _, _ = val_metrics(model, data, dev, amp, bs_eval)
                rec = {"iter": done, "lr": lr, "val_acc": acc, "in_window": in_window}
                if in_window and acc > best["acc"]:
                    best = {"acc": acc, "iter": done}
                    atomic_save({"model": model.state_dict(), "iter": done, "val_acc": acc}, best_p)
                    rec["best"] = True
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[{name}] validation at iteration {done}: accuracy {acc:.4f}"
                      + (" (best within the selection window)" if rec.get("best") else ""), flush=True)
            if done % val_every == 0:
                save_state(done)
    except KeyboardInterrupt:
        save_state(completed)
        print(f"[{name}] stopped on request after iteration {completed}; rerun the same command to resume", flush=True)
        raise SystemExit(130)
    finally:
        signal.signal(signal.SIGINT, old)
    if stop_at < total:                                      # branch point: save a separate copy; the run is not finished
        save_state(stop_at)
        tmp = out / f"state_{stop_at}.pt.tmp"
        tmp.write_bytes(last_p.read_bytes())
        os.replace(tmp, out / f"state_{stop_at}.pt")
        (out / "paused.json").write_text(json.dumps({"run": name, "iter": stop_at, "total": total,
                                                     "time": time.strftime("%Y-%m-%d %H:%M:%S")}), encoding="utf-8")
        print(f"[{name}] paused at iteration {stop_at}: state_{stop_at}.pt saved", flush=True)
        return
    save_state(total)
    st = torch.load(best_p, map_location=dev, weights_only=False)
    model.load_state_dict(st["model"])
    _, vc, vs = val_metrics(model, data, dev, amp, bs_eval)
    tc = score(model, data.cover, data.test, dev, amp, bs_eval)
    ts = score(model, data.stego, data.test, dev, amp, bs_eval)
    np.savez_compressed(out / "scores.npz", val_ids=np.array([data.ids[p] for p in data.val]), val_cover=vc, val_stego=vs,
                        test_ids=np.array([data.ids[p] for p in data.test]), test_cover=tc, test_stego=ts)
    info = {"run": name, "config": run, "init_resolved": init, "phases": phases, "iterations": total, "best_iter": st["iter"],
            "best_val_acc": st["val_acc"], "n_train": int(len(data.train)), "n_val": int(len(data.val)),
            "n_test": int(len(data.test)), "smoke": smoke, "amp": amp, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "config_sha256": hashlib.sha256(CFG_P.read_bytes()).hexdigest(),
            "data_split_sha256": data.index.get("split_sha256"), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "done.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[{name}] done: snapshot at iteration {st['iter']}, validation accuracy {st['val_acc']:.4f}", flush=True)


def plan(cfg: dict, its: float | None) -> None:
    """Schedule: common G4 200k iterations, short branch 50k; four confirmatory runs. The 'fallback' line runs only if the short G4 fails."""
    def h(k):
        return f", about {k / its / 3600:.1f} h" if its else ""
    full = cfg["runs"]["g4_hill_0.4"]
    b = full["branch_at"]
    short = sum(p[0] for p in cfg["runs"]["g4_hill_0.4_short"]["phases"]) - b
    rows = [("g4", "g4_hill_0.4 (to branch point)", b, "random"), ("g4", "g4_hill_0.4_short", short, "iteration 200000")]
    rows += [("confirm", n, sum(p[0] for p in cfg["runs"][n]["phases"]), cfg["runs"][n]["init"]) for n in cfg["sequences"]["confirm"]]
    tot = 0
    for seq, n, k, init in rows:
        tot += k
        print(f"{seq:8s} {n:28s} {k:7d} iterations, initialization {init:16s}{h(k)}")
    rest = sum(p[0] for p in full["phases"]) - b
    print(f"total {tot} iterations{h(tot)}; fallback (short G4 fails): {rest} more iterations of g4_hill_0.4{h(rest)}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--sequence", choices=["confirm"])
    g.add_argument("--plan", action="store_true", help="print the training schedule; with --its also the estimated time")
    ap.add_argument("--its", type=float, default=None, help="measured speed (iterations/s, batch of 16 pairs of 256x256)")
    ap.add_argument("--until", type=int, default=None, help="stop at this iteration and save state_<iteration>.pt (branch point)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="code test only: 20-iteration schedule, written to results/srnet_smoke/")
    a = ap.parse_args()
    cfg = json.loads(CFG_P.read_text(encoding="utf-8"))
    if a.plan:
        plan(cfg, a.its)
        return
    check_frozen()
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if a.device == "cuda" and dev.type != "cuda":
        raise SystemExit("no CUDA GPU found: check .venv-gpu (torch +cu126 build) and the driver")
    torch.backends.cudnn.benchmark = True
    out_root = ROOT / "results" / ("srnet_smoke" if a.smoke else "srnet")
    names = [a.run] if a.run else cfg["sequences"][a.sequence]
    for n in names:
        if n.startswith("confirm") and not a.smoke:
            s = ROOT / "results" / "srnet" / "g4_srnet_summary.json"
            if not s.exists() or not json.loads(s.read_text(encoding="utf-8")).get("G4_SRNet_pass"):
                raise SystemExit("SRNet G4 has not passed (results/srnet/g4_srnet_summary.json): the confirmatory step is not run")
        run_one(cfg, n, dev, out_root, a.smoke, a.until)


if __name__ == "__main__":
    main()
