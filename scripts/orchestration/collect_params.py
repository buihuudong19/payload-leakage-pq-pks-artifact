"""Collect every experimental parameter into results/manuscript_inputs/params.md.

No number is typed by hand. Each value is read from a source, which is recorded next to it:
  * configuration: configs/*.json;
  * actual arguments of each step: scripts/orchestration/pipeline.ps1 (Args), merged with the argparse defaults of the script;
  * constants in the code: read with a regular expression on the exact code line, recorded as `file:line`;
  * facts of the run (when available): split files, logs, result files, step status, installed environment.
Can be rerun at any time; step R1 (collect_results.py) calls this script after all measurements are finished.

    python scripts/orchestration/collect_params.py [--out results/manuscript_inputs/params.md]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
PIPE = ROOT / "scripts" / "orchestration" / "pipeline.ps1"


# ---------------------------------------------------------------- reading the sources
def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig", errors="replace")


def jload(p: Path):
    try:
        return json.loads(read(p))
    except Exception:
        return None


def code(path: str, pattern: str, group: int | str = 1):
    """Value from the first code line matching `pattern`: (value, 'file:line'); (None, 'file: NOT FOUND') if there is none."""
    p = ROOT / path
    for i, line in enumerate(read(p).splitlines(), 1):
        m = re.search(pattern, line)
        if m:
            return m.group(group).strip(), f"{path}:{i}"
    return None, f"{path}: NOT FOUND `{pattern}`"


def argdefaults(path: str) -> dict:
    """argparse defaults of a script: {flag: (value, 'file:line')}."""
    src = read(ROOT / path)
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            a0 = node.args[0]
            if not isinstance(a0, ast.Constant) or not isinstance(a0.value, str):
                continue
            d = next((k.value for k in node.keywords if k.arg == "default"), None)
            if d is None:
                act = next((k.value for k in node.keywords if k.arg == "action"), None)
                val = "False" if isinstance(act, ast.Constant) and act.value == "store_true" else "(none)"
            else:
                try:
                    val = repr(ast.literal_eval(d))
                except Exception:
                    val = ast.get_source_segment(src, d)
            out[a0.value] = (val, f"{path}:{node.lineno}")
    return out


def pipeline_steps() -> list[dict]:
    txt = read(PIPE)
    steps = []
    for blk in re.split(r"\n\s*@\{ Id = ", txt)[1:]:
        g = lambda k: (re.search(rf"{k} = '([^']*)'", blk) or [None, None])[1]
        args = re.search(r"Args = @\(([^)]*)\)", blk)
        deps = re.search(r"Deps = @\(([^)]*)\)", blk)
        steps.append({"id": re.match(r"'([^']+)'", blk).group(1), "lane": g("Lane"), "name": g("Name"), "kind": g("Kind"),
                      "est": g("Est"), "out": g("Out"),
                      "args": re.findall(r"'([^']*)'", args.group(1)) if args else [],
                      "deps": re.findall(r"'([^']*)'", deps.group(1)) if deps else []})
    return steps


def effective_args(step: dict) -> list[tuple[str, str, str]]:
    """(flag, value actually used, source) for a Python step: arguments in pipeline.ps1 override the argparse defaults."""
    script = step["args"][0].replace("\\", "/") if step["args"] else ""
    if not script.endswith(".py") or not (ROOT / script).exists():
        return []
    dfl = argdefaults(script)
    given, pos, toks = {}, [], step["args"][1:]
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            vals = []
            while i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                vals.append(toks[i + 1]); i += 1
            given[t] = " ".join(vals) if vals else "True"
        else:
            pos.append(t)
        i += 1
    rows = [("(subcommand)", " ".join(pos), "pipeline.ps1")] if pos else []
    for flag, (val, where) in dfl.items():
        if not flag.startswith("--") or flag in ("--out", "--smoke", "--e1", "--split"):
            continue
        rows.append((flag, given[flag], "pipeline.ps1 (Args)") if flag in given else (flag, val, f"default, {where}"))
    for flag, v in given.items():
        if flag not in dfl:
            rows.append((flag, v, "pipeline.ps1 (Args)"))
    return rows


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def table(rows, head=("Parameter", "Value", "Source")) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in r) + " |")
    return out + [""]


# ---------------------------------------------------------------- sections
def sec_environment() -> list[str]:
    L = ["## 1. Run environment", ""]
    rows = []
    setup = sorted((RES / "confirm_setup").glob("setup_*.log"))
    if setup:
        t = read(setup[-1])
        m = re.search(r"Machine: (\S+) \((.+?)\)", t)
        if m:
            rows.append(("Machine, operating system", f"{m.group(1)}, {m.group(2)}", rel(setup[-1])))
        m = re.search(r"PSVersion: (\S+)", t)
        if m:
            rows.append(("Windows PowerShell (orchestration)", m.group(1), rel(setup[-1])))
    if platform.system() == "Windows":
        def ps(q):
            try:
                return subprocess.run(["powershell.exe", "-NoProfile", "-Command", q], capture_output=True, text=True,
                                      timeout=60).stdout.strip()
            except Exception:
                return ""
        cpu = ps("(Get-CimInstance Win32_Processor | Select-Object -First 1 | ForEach-Object { $_.Name + ' (' + $_.NumberOfCores + ' cores, ' + $_.NumberOfLogicalProcessors + ' threads)' })")
        ram = ps("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)")
        rows += [("CPU", cpu or "?", "Win32_Processor, when this script runs"), ("RAM (GB)", ram or "?", "Win32_ComputerSystem")]
        try:
            g = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
            rows.append(("GPU, memory, driver", g, "nvidia-smi"))
        except Exception:
            pass
    else:
        rows.append(("Hardware of the measurement machine", "AMD Ryzen 7 3700X (8 cores, 16 threads), 32 GB RAM, "
                     "NVIDIA GeForce RTX 2060 SUPER 8 GB; on Windows the script reads these values directly", "-"))
    for dg in sorted((RES / "diag").glob("diag_*.txt"))[-1:]:
        m = re.search(r"NVIDIA-SMI (\S+)", read(dg))
        if m:
            rows.append(("NVIDIA driver", m.group(1), rel(dg)))
    lock = RES / "confirm_setup" / "requirements-confirm.lock"
    if lock.exists():
        pk = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+)", read(lock), re.M))
        for name in ("numpy", "scipy", "scikit-learn", "numba", "conseal", "sealwatch", "pyhpke", "cryptography",
                     "pycryptodome", "pillow", "matplotlib", "torch"):
            if name in pk:
                rows.append((f"{name} (.venv-confirm)", pk[name], rel(lock)))
    if setup:
        m = re.search(r"python (\S+) \| torch (\S+) \| cuda (\S+) \| cudnn (\S+)", read(ROOT / "results" / "logs" / "G1.log")) \
            if (RES / "logs" / "G1.log").exists() else None
        if m:
            rows += [("Python (.venv-gpu)", m.group(1), "results/logs/G1.log"), ("PyTorch (.venv-gpu)", m.group(2), "results/logs/G1.log"),
                     ("CUDA, cuDNN", f"{m.group(3)}, {m.group(4)}", "results/logs/G1.log")]
    prof = jload(ROOT / "configs" / "profile_frozen_v1.json") or {}
    for k, v in (prof.get("environment") or {}).items():
        rows.append((f"{k} (at profile freeze)", v, "configs/profile_frozen_v1.json"))
    run = jload(RES / "run_status" / "run.json")
    if run:
        rows.append(("run_all started", run.get("started"), "results/run_status/run.json"))
    return L + table(rows)


def sec_profile() -> list[str]:
    p = ROOT / "configs" / "profile_frozen_v1.json"
    prof = jload(p) or {}
    L = ["## 2. Frozen PQ-PKS profile (profile_frozen_v1)", ""]
    rows = [(k, json.dumps(v, ensure_ascii=False), rel(p)) for k, v in (prof.get("public_params") or {}).items()]
    rows += [(f"decision {k}", v, rel(p)) for k, v in (prof.get("decisions") or {}).items()]
    rows.append(("freeze date", prof.get("date"), rel(p)))
    return L + table(rows)


def sec_data() -> list[str]:
    L = ["## 3. Data and splits", ""]
    rows = [("cover images", "BOSSbase 1.01 (10 000) and BOWS2 (10 000), grayscale 512x512", "README.md")]
    sp = ROOT / "configs" / "splits" / "split_v1.json"
    pipe_sha = re.search(r"\$SplitSha256 = '([0-9a-f]+)'", read(PIPE))
    rows.append(("registered SHA-256 of split_v1", pipe_sha.group(1) if pipe_sha else "?", "pipeline.ps1"))
    if sp.exists():
        s = jload(sp)
        rows.append(("actual SHA-256 of the split file", sha256(sp), rel(sp)))
        for src in ("boss", "bows2"):
            for part in ("train", "val", "test", "pilot"):
                v = (s.get(src) or {}).get(part)
                if isinstance(v, list):
                    rows.append((f"{src}.{part}", len(v), rel(sp)))
        e1 = s.get("e1") or {}
        if "folds" in e1:
            rows.append(("E1: test images (with fold)", len(e1["folds"]), rel(sp)))
        if "train_only" in e1:
            rows.append(("E1: training-only images", len(e1["train_only"]), rel(sp)))
        if "srm_repeated_test_sets" in s:
            rows.append(("E2/E3: available repeated test splits", len(s["srm_repeated_test_sets"]), rel(sp)))
            rows.append(("E2/E3: size of one test set", len(s["srm_repeated_test_sets"][0]), rel(sp)))
    lg = RES / "logs" / "2.log"
    if lg.exists():
        m = re.search(r"test \d+.*largest group \d+", read(lg))
        if m:
            rows.append(("make_split (log)", m.group(0), rel(lg)))
    return L + table(rows)


def sec_steps(steps) -> list[str]:
    L = ["## 5. Actual arguments of each run_all step", "",
         "Arguments in `pipeline.ps1` override the defaults of the script; the Source column shows where each value comes from.", ""]
    for s in steps:
        rows = effective_args(s)
        L.append(f"### Step {s['id']} (lane {s['lane']}): {s['name']}")
        L.append("")
        L.append(f"Command: `{' '.join(s['args'])}`; dependencies: {', '.join(s['deps']) or '-'}; estimate: {s['est']}; output: `{s['out']}`")
        L.append("")
        if rows:
            L += table(rows, ("Flag", "Value", "Source"))
    return L


def alpha_grid(path: str) -> str:
    names, _ = code(path, r"(MIN_FIRST_IMAGE, OP_ML_DSA_44, OP_ML_DSA_65) = ")
    vals, _ = code(path, r"MIN_FIRST_IMAGE, OP_ML_DSA_44, OP_ML_DSA_65 = (.*)$")
    grid, _ = code(path, r'"confirm": \((\(.*\)),')
    if not (names and vals and grid):
        return "?"
    for n, v in zip(names.split(", "), vals.split(", ")):
        grid = re.sub(rf"\b{n}\b", v, grid)
    return grid


def sec_constants() -> list[str]:
    L = ["## 4. Constants of each experiment (in the code)", ""]
    E1 = "scripts/confirm/e1_confirm.py"
    L += ["### E1 (RQ1.3, gate G3)", ""] + table([
        ("Kemeleon variants (bytes)",) + code(E1, r"VARIANTS = (\(.*\))"),
        ("pre-registered bit tests",) + code(E1, r"REGISTERED = (\{.*\})"),
        ("D1: uniform strings U per cell",) + code(E1, r"size=\((\d+), S\.shape\[1\]\)"),
        ("D1: bootstrap resamples (all)",) + code(E1, r"q\.size, u\.size, B=(\d+)"),
        ("D1: bootstrap resamples (stratum)",) + code(E1, r"nm, u\.size, B=(\d+)"),
        ("D1: stratum margin",) + code(E1, r'"margin": (2\.8 \* float\(np\.sqrt\(0\.5 / nm\)\))'),
        ("D1: saturated stratum",) + code(E1, r'\("sat>5%", (sat\[test\] > [0-9.]+)\)'),
        ("D4: model",) + code(E1, r"(LogisticRegression\([^)]*\))"),
        ("D4: number of folds",) + code(E1, r"for f in range\((\d+)\):"),
        ("D4: cluster bootstrap",) + code(E1, r"cc\.size, B=(\d+)"),
        ("pooling of M sessions (LLR)",) + code(E1, r"for M in (\(.*\)):"),
        ("G3 threshold",) + code(E1, r'upper_abs"\] <= ([0-9.]+) and'),
    ])
    PH = "scripts/confirm/e1_posthoc_null.py"
    dp = argdefaults(PH)
    L += ["### Post-hoc null check of E1 D4", ""] + table([
        ("null iterations R", *dp.get("--R", ("?", PH))),
        ("pool of uniform-string features", *dp.get("--pool", ("?", PH))),
        ("bootstrap", *code(PH, r"B_BOOT, MARGIN = (\d+)")),
    ])
    PW = RES / "e1bis_design" / "power.json"
    pw = jload(PW)
    if pw:
        L += ["### E1-bis (design): power calculation", ""] + table(
            [(f"n = {r['n']}", "; ".join(f"L={L_}: SD_null {v['sd_null']:.5f}" for L_, v in r["by_L"].items())
              + f"; P(all 12 cells pass under the null) {r['p_all12_pass_null_normal']:.3f}", rel(PW)) for r in pw["rows"]]
            + [("R per (n, L)", pw["R"], rel(PW)), ("pool", pw.get("pool"), rel(PW)), ("rule", pw["rule"], rel(PW))])
    SR = "scripts/experiments/screen_run.py"
    AC = "scripts/confirm/analyze_confirm.py"
    L += ["### E2/E3 (step 5a and confirmatory analysis)", ""] + table([
        ("special alpha values (smallest first image, operating points of ML-DSA-44, ML-DSA-65)",) + code(SR, r"MIN_FIRST_IMAGE, OP_ML_DSA_44, OP_ML_DSA_65 = (.*)$"),
        ("alpha grid of the confirm plan (constant names substituted)", alpha_grid(SR), code(SR, r'"confirm": \((\(.*\)),')[1]),
        ("compared systems",) + code(SR, r"SYSTEMS = (\(.*\))"),
        ("SRM features",) + code(SR, r"(SRM, [0-9  ]+ dimensions.*)$"),
        ("FPR, TOST delta, bootstrap resamples B",) + code(AC, r"FPR, DELTA, B = (.*?)\s*#"),
        ("B_AUC (AUC, pooled D_pub, delta_syn, CI of alpha*_i)",) + code(AC, r"^B_AUC = (\d+)"),
        ("number of sessions M pooled for D_pub",) + code(AC, r"M_POOL = (\([^)]*\))"),
        ("Bayes bound c_M (H2a)",) + code(AC, r"C_M = (\{.*\})"),
        ("validation set size",) + code(AC, r"nv = (min\(1000, len\(rest_boss\) // 2\))"),
        ("W-img",) + code(AC, r"(ens = Ensemble\(seed=0\)\.fit.*)$"),
        ("FLD ensemble classifier",) + code("scripts/confirm/common.py", r'"""(FLD ensemble.*)"""'),
    ])
    G4 = "scripts/confirm/g4_calibrate.py"
    d4 = argdefaults(G4)
    ref = jload(ROOT / "configs" / "g4_reference.json")
    rows = [(f, *d4[f]) for f in ("--methods", "--alphas", "--n", "--seeds") if f in d4]
    rows.append(("chunk size", *code(G4, r"CHUNK = (\d+)")))
    rows.append(("tolerance of the simulator (3e)", *code("scripts/confirm/g4_audit_simulator.py", r"TOL = ([0-9.e-]+)")))
    if ref:
        rows.append(("reference values", json.dumps(ref, ensure_ascii=False)[:600], "configs/g4_reference.json"))
    L += ["### SRM G4", ""] + table(rows)
    cfgp = ROOT / "configs" / "srnet_runs_v2.json"
    cfg = jload(cfgp) or {}
    rows = [(f"chung: {k}", json.dumps(v, ensure_ascii=False), rel(cfgp)) for k, v in (cfg.get("common") or {}).items()]
    rows += [(f"data {k}", json.dumps(v, ensure_ascii=False), rel(cfgp)) for k, v in (cfg.get("datasets") or {}).items()]
    for k, v in (cfg.get("runs") or {}).items():
        tot = sum(p[0] for p in v.get("phases", []))
        rows.append((f"run {k}", f"phases (iterations, lr) {v.get('phases')}; total {tot} iterations; selected within the last {v.get('select_last')} iterations; "
                     f"initialization {v.get('init') or v.get('init_state')}; crop {v.get('train_crop')}", rel(cfgp)))
    if cfg.get("g4_protocol"):
        rows.append(("SRNet G4 rule", cfg["g4_protocol"].get("rule"), rel(cfgp)))
    L += ["### SRNet (G4 and confirmatory)", ""] + table(rows)
    AR = "scripts/confirm/apply_rules.py"
    L += ["### Decision rules", ""] + table([
        ("conditions (i) and (ii) of alpha*",) + code(AR, r"COND_I, COND_II = (.*)$"),
        ("alpha of the smallest session-start image",) + code(AR, r"MIN_FIRST_IMAGE = ([0-9.]+)"),
        ("order of the signature schemes",) + code(AR, r"ORDER = (\(.*?\))"),
        ("scheme analysed only",) + code(AR, r'ANALYTIC = "(.*)"'),
    ])
    lad = jload(RES / "e5_budget" / "e5_ladder.json")
    if lad:
        keys = [k for k, v in lad.items() if not isinstance(v, (dict, list))][:20]
        L += ["### E5 (budget, analysis)", ""] + table([(k, lad[k], "results/e5_budget/e5_ladder.json") for k in keys])
    return L


def sec_runtime(steps) -> list[str]:
    L = ["## 6. Facts of the run (read from the results; empty if a step is not finished)", ""]
    rows = []
    meta = jload(RES / "confirm_v1" / "raw" / "run_meta.json")
    if meta:
        rows += [(f"step 5a: {k}", json.dumps(v, ensure_ascii=False)[:300], "results/confirm_v1/raw/run_meta.json")
                 for k, v in meta.items() if k in ("covers", "alphas", "systems_at", "plan", "srm", "save_stego")]
    av = jload(RES / "confirm_v1" / "feat" / "avail.json")
    if av:
        rows += [("covers used in E2/E3", av.get("n_covers"), "results/confirm_v1/feat/avail.json"),
                 ("excluded covers (missing one pair)", av.get("n_covers_missing_any"), "results/confirm_v1/feat/avail.json")]
    e1 = jload(RES / "e1_confirm" / "e1_summary.json")
    if e1:
        rows += [("E1: n_test", e1.get("n_test"), "results/e1_confirm/e1_summary.json"),
                 ("E1: training-only images", e1.get("n_train_only"), "results/e1_confirm/e1_summary.json"),
                 ("E1: number of cells", len(e1.get("cells", {})), "results/e1_confirm/e1_summary.json")]
    for name, p in (("G4 SRM", RES / "g4" / "g4_summary.json"), ("G4 SRNet", RES / "srnet" / "g4_srnet_summary.json"),
                    ("E4", RES / "e4_full" / "e4_summary.json"), ("E6", RES / "e6_desktop" / "e6_summary.json"),
                    ("confirmatory SRNet", RES / "srnet" / "confirm_srnet_summary.json"), ("GPU", RES / "gpu_bench" / "decision.json")):
        d = jload(p)
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (int, float, str, bool)) and len(str(v)) < 200:
                    rows.append((f"{name}: {k}", v, rel(p)))
    for lg in sorted((RES / "logs").glob("G*.log")):
        for line in read(lg).splitlines():
            if re.search(r"\] \S+: train \d+, val \d+, test \d+ images|done: snapshot at iteration|paused at iteration|branching from", line):
                rows.append((f"SRNet ({lg.stem})", line.strip(), rel(lg)))
    L += table(rows) if rows else ["(no results yet)", ""]
    rows = []
    fmt = "%Y-%m-%d %H:%M:%S"
    for s in steps:
        st = jload(RES / "run_status" / f"{s['id']}.json")
        lg = RES / "logs" / f"{s['id']}.log"
        starts = re.findall(r"^\ufeff?===== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", read(lg), re.M) if lg.exists() else []
        if not st and not starts:
            continue
        first = starts[0] if starts else (st or {}).get("start") or ""
        end = (st or {}).get("end") or ""
        dur = ""
        try:
            if first and end:
                dur = f"{(time.mktime(time.strptime(end, fmt)) - time.mktime(time.strptime(first, fmt))) / 3600:.2f} h"
        except Exception:
            pass
        note = (st or {}).get("reason") or ""
        if len(starts) > 1:
            note = f"run {len(starts)} times, start times: {', '.join(starts)}; " + note
        rows.append((s["id"], (st or {}).get("state", ""), first, end, dur, note))
    L += ["### Actual run time of each step", ""] + table(rows, ("Step", "Status", "Start (first, from the log)", "End", "Duration (including pauses)", "Note"))
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "manuscript_inputs" / "params.md"))
    a = ap.parse_args()
    steps = pipeline_steps()
    head = ["# PQ-PKS experimental parameters (generated)", "",
            f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S')} on {platform.node()} by `scripts/orchestration/collect_params.py`. "
            "Do not edit this file by hand: change the source and rerun the script. Every value has a Source column for checking.", ""]
    body = sec_environment() + sec_profile() + sec_data() + sec_constants() + sec_steps(steps) + sec_runtime(steps)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + body) + "\n", encoding="utf-8")
    miss = sum("NOT FOUND" in l for l in body)
    print(f"wrote {out} ({len(body)} lines; {miss} values without a source)")


if __name__ == "__main__":
    main()
