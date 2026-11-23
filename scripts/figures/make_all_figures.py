"""Regenerate every figure of the PQ-PKS paper; optionally copy them into a manuscript directory (--manuscript).

    python scripts/figures/make_all_figures.py            (or make_all_figures.sh / make_all_figures.cmd)

Steps (a step whose data are missing is skipped and reported):
  1. Figs. 1, 3, 4 (diagrams, no data needed)                       -> results/figures/
  2. Figs. 2, 6-11 screening version and tables (make_figures.py)  -> results/figures/
     (the file names keep the earlier numbering fig5_...fig10_; the figure numbers in the paper are one higher)
  2b. Fig. 5 "one cover, end to end" (make_fig_example.py): redrawn from results/figures/fig5_one_cover_data.npz if present
     (exactly the published figure), otherwise embedded again (needs data/external/bossbase/8811.pgm)
  3. Confirmatory Fig. 5 (make_e1_confirm.py), if results/e1_confirm/e1_summary.json exists      -> results/figures_confirm/
  4. Confirmatory Figs. 6-8 (make_figures_confirm.py), if results/confirm_v1/summary.json exists -> results/figures_confirm/
  5. With --manuscript DIR: copy the figures listed in FIGS (.pdf and .png) into DIR/figures/.
Common figure style: ieee_style.py.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Names of the figures used in the paper. Only these files are copied, so stale files in results/ never reach the manuscript.
FIGS = [
    "fig1_overview", "fig2_prop1_llr", "fig3_architecture", "fig4_flows", "fig5_one_cover", "fig5_e1_syndrome",
    "fig6_dpub_vs_M", "fig7_tpr_wardens", "fig8_pe_alpha_tost", "fig9_bits_vs_N", "fig10_bit_budget",
    # confirmatory version (only once the confirmatory results exist)
    "fig5_e1_confirm", "fig6_dpub_vs_M_confirm_a04", "fig7_tpr_wardens_confirm",
    "fig7_tpr_wardens_confirm_a04", "fig7_tpr_wardens_confirm_a02", "fig8_pe_alpha_tost_confirm",
    "fig9_bits_vs_N_confirm", "fig10_bit_budget_confirm",
]


def run(args: list[str]) -> bool:
    print(">>", " ".join(Path(a).name if a.endswith(".py") else a for a in args), flush=True)
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    if r.returncode:
        print(f"   error (code {r.returncode})", flush=True)
    return r.returncode == 0


def manuscript_dir(arg: str | None) -> Path | None:
    return Path(arg) if arg else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=None, help="manuscript directory; figures are copied into its figures/ subdirectory")
    ap.add_argument("--no-copy", action="store_true")
    a = ap.parse_args()
    fig, figc = ROOT / "results" / "figures", ROOT / "results" / "figures_confirm"
    ok = run([str(HERE / "fig_diagrams.py"), "--out", str(fig)])
    ok &= run([str(ROOT / "scripts" / "experiments" / "make_figures.py"), "--out", str(fig)])
    ex_data = fig / "fig5_one_cover_data.npz"
    ok &= run([str(HERE / "make_fig_example.py"), "--out", str(fig)] + (["--data", str(ex_data)] if ex_data.exists() else []))
    if (ROOT / "results" / "e1_confirm" / "e1_summary.json").exists():
        ok &= run([str(ROOT / "scripts" / "experiments" / "make_e1_confirm.py"), "--out", str(figc)])
    else:
        print("-- confirmatory Fig. 5 skipped: results/e1_confirm/e1_summary.json not found")
    if (ROOT / "results" / "confirm_v1" / "summary.json").exists():
        ok &= run([str(ROOT / "scripts" / "experiments" / "make_figures_confirm.py"), "--out", str(figc)])
    else:
        print("-- confirmatory Figs. 6-8 skipped: results/confirm_v1/summary.json not found")
    md = None if a.no_copy else manuscript_dir(a.manuscript)
    if md is not None:
        dst = md / "figures"
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for name in FIGS:
            for ext in (".pdf", ".png"):
                for src in (figc, fig):
                    f = src / (name + ext)
                    if f.exists():
                        shutil.copy2(f, dst / f.name)
                        n += 1
                        break
        print(f"copied {n} figure files to {dst}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
