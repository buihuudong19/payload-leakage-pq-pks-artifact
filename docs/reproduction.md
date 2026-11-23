# Reproducing the experiments

This guide describes how the results of the article were produced and how to produce them again. The confirmatory
step ran on one Windows 11 desktop (AMD Ryzen 7 3700X, 8 cores and 16 threads, 32 GB RAM, NVIDIA GeForce RTX 2060
SUPER 8 GB) and took several days of computation; the individual scripts also run on Linux.

## 1. Environments

| Environment | Created by | Content | Used for |
|---|---|---|---|
| `.venv-confirm` | `scripts\confirm\setup_confirm.ps1` | Python 3.12 (managed by uv), `requirements.txt`, `sealwatch==2026.6`, `scikit-learn==1.9.1`, `matplotlib`, this package in editable mode | every CPU step |
| `.venv-gpu` | `scripts\gpu\setup_gpu.ps1` | Python 3.12, PyTorch 2.8.0 with CUDA 12.6, numpy, pillow (`scripts/gpu/requirements-gpu.txt`) | SRNet training and benchmark |

Both scripts keep everything inside the repository (`.tools\`, `.venv-confirm\`, `.venv-gpu\`), do not change the
system Python or `PATH`, and record the installed versions (`results\confirm_setup\requirements-confirm.lock`,
`scripts\gpu\requirements-gpu.lock`). `setup_confirm.ps1` also runs the test suite and `freeze_profile.py --check`.

On Linux the equivalent CPU environment is

```bash
python3.12 -m venv .venv-confirm && source .venv-confirm/bin/activate
pip install -r requirements.txt sealwatch==2026.6 scikit-learn==1.9.1 matplotlib
pip install -e .
```

## 2. Data

Prepare BOSSbase 1.01, BOWS2-Original and ALASKA2 as described in [`data.md`](data.md). The default locations are
`data/external/bossbase/`, `data/external/bows2/` and `data/external/alaska2/pgm/`; `PQPKS_BOSSBASE_DIR`,
`PQPKS_BOWS2_DIR` and `PQPKS_ALASKA2_DIR` override them.

## 3. Checks before running

```bash
python scripts/freeze_profile.py --check     # src/ and configs/ match the frozen profile (see provenance.md)
python scripts/setup/fetch_third_party.py    # module of [Zha25], needed by BL4 (ablation runs and tests)
pytest -q
```

## 4. Full confirmatory run (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -DryRun   # about 2 minutes, runs no command
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Reset    # full run
powershell -ExecutionPolicy Bypass -File .\run_all.ps1           # resume after an interruption
```

`run_all.ps1` checks that the BOSSbase and BOWS2 folders exist and that the drive has at least 250 GB free (stego
features of step C4 about 110 GB, memmaps about 58 GB, SRNet data about 30 GB), then opens one window per automatic lane
(W1 to W9) and a monitor window. The steps, their lanes, dependencies, estimated durations and outputs are listed in
[`steps.md`](steps.md). Lanes W10 and W11 (phase E and the final aggregation) and W12 (post-hoc analyses, which change
no registered decision) are opened by hand:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\orchestration\run_lane.ps1 -Lane W10
powershell -ExecutionPolicy Bypass -File scripts\orchestration\run_lane.ps1 -Lane W11
powershell -ExecutionPolicy Bypass -File scripts\orchestration\run_lane.ps1 -Lane W12
```

Status: `results\run_status\STATUS.md` (rewritten by the monitor at every refresh). Logs: `results\logs\<step>.log`.
A step interrupted by a closed window or a power loss stays in state `running`; rerunning `run_all.ps1` starts it
again, and the long scripts (C1, C4, C5, 3a, 3e, SRNet) resume from the part already finished. `run_lane.ps1 -Lane <L>
-Only <steps>` runs selected steps of a lane in a separate window.

Steps whose output has a registered SHA-256 (`split_v1.json`, `e1bis_ids.txt`, `e1bis_v1.json`, `e7_v1.json`) fail if
the regenerated file differs from the registered one; nothing downstream runs in that case.

## 5. Running steps by hand

Every step of [`steps.md`](steps.md) is a single command and can be run on its own, in the order of the dependencies,
with the Python of the corresponding environment. For example, on Linux:

```bash
export PQPKS_BOSSBASE_DIR=data/external/bossbase PQPKS_BOWS2_DIR=data/external/bows2
python scripts/confirm/make_split.py --workers 8                      # must reproduce configs/splits/split_v1.json
python scripts/confirm/e1_confirm.py extract --workers 8
python scripts/confirm/e1_confirm.py analyze                          # E1 -> results/e1_confirm/e1_summary.json
python scripts/experiments/screen_run.py --plan confirm --srm --workers 6 --out results/confirm_v1/raw --save-stego all
python scripts/confirm/analyze_confirm.py consolidate
python scripts/confirm/analyze_confirm.py analyze --reps 10 --out summary_r10.json
python scripts/confirm/promote_r10.py
python scripts/confirm/apply_rules.py --no-srnet                      # alpha* and the decision rules
python scripts/experiments/write_results_confirm.py                   # results/RESULTS_TABLES.md
```

`make_split.py` refuses to overwrite an existing split file; the frozen file in `configs/splits/` is the one used in the
article.

## 6. Outputs and the article

| Output | Produced by | Content |
|---|---|---|
| `results/e1_confirm/e1_summary.json`, `results/e1bis/e1bis_summary.json` | C3, X5 | E1 and E1-bis (gates G3 and G3') |
| `results/confirm_v1/summary.json`, `decisions.json` | C7/PF1-PF3, C8 | E2/E3 with SRM: P_E, TPR, TOST, D_pub, alpha*, decision rules |
| `results/srnet/confirm_srnet_summary.json`, `decisions_srnet.json` | G12, G13 | SRNet results and the decision rules with SRNet |
| `results/e4_full/e4_summary.json` | E4 | functional block on 512x512 images |
| `results/e5_budget/`, `results/e6_desktop/e6_summary.json` | C10/PF4, C9 | capacity, bit budget and timing |
| `results/e7/`, `ablate/`, `ablate2/`, `ra/`, `jpeg/`, `jpeg95/`, `confirm_r/` | D and PE steps | E7 and the supplementary experiments |
| `results/quality/` | `scripts/analysis/quality_metrics.py` | PSNR and SSIM of the stego images |
| `results/RESULTS_TABLES.md`, `results/phase_d/PHASE_D_TABLES_en.md` | R0/PF6, PF5 | tables of the article and of the supplementary material |
| `results/figures/`, `results/figures_confirm/` | PF8 (`scripts/figures/make_all_figures.py`) | figures |

These outputs, together with the per-cell warden scores and the extracted strings, are archived in the dataset record
`pq-pks-raw-data-v1` (https://doi.org/10.5281/zenodo.24462430). The tables and figures can be regenerated from its core archive
(`pq-pks-raw-data-v1-core.zip`) alone, without rerunning the measurements: copy its `results/` folder into the
repository and run steps R0, PF5 and PF8. The other two archives (`-strings.zip`, `-e1-syndromes.zip`) are needed only to
rerun the payload distinguishers on the extracted strings and the E1/E1-bis analyses.

## 7. Scripts outside the confirmatory step

| Script | Purpose |
|---|---|
| `scripts/theory/verify_constants.py` | recomputes every analytic constant of the article with exact arithmetic |
| `scripts/prop1_mlkem_distinguisher.py` | Proposition 1: raw ML-KEM-768 ciphertexts against uniform strings and Kemeleon encodings |
| `scripts/build_stc_table.py` | rebuilds the public STC table `src/pqpks/stego/stc_table_h10.json` (and the h = 8, 12 tables of ablation A9) |
| `scripts/sanity_matrix.py` | cross-check matrix of the extracted strings of every system before E2 |
| `scripts/demo_pqpks_flow.py` | a three-image PQ-PKS session with negative cases (synthetic test covers) |
| `scripts/pilot_baselines.py`, `scripts/pilot_report.py`, `scripts/pilot_review_v3.py` | pilot rounds on 500-1000 BOSSbase images, including the layout pilot of Section 5 (A8 against the fixed layout A8') |
| `scripts/experiments/screen_run.py --plan screen`, `scripts/experiments/screen_analyze.py` | screening round on the 2000 pilot and screening covers |
| `scripts/analysis/budget_v2.py` | bit budget by region, bucket thresholds, receiver trials |
| `scripts/analysis/alpha_star_thresholds.py` | post-hoc sensitivity of alpha* to the threshold of condition (i), from `results/confirm_v1/summary.json` and `results/srnet/decisions_srnet.json` (Table S12 of the supplementary material; changes no registered decision; step PH3) |
| `scripts/confirm/e1_posthoc_null.py` | post-hoc check of the learned distinguisher D4 of E1 under the uniform null: null SD, coverage of the registered interval, null p-values (step PH1; G3 stays failed) |
| `scripts/confirm/e1_posthoc_refit.py` | post-hoc interval of D4 that refits the five models in every bootstrap replicate, and its coverage under the uniform null (`--quick` gives the run time; step PH2; G3 stays failed) |

## 8. Determinism

The analysis steps are deterministic given their inputs: the splits are frozen, and the FLD ensemble (seed 0), the SPAM
random generator and the bootstrap use fixed seeds, so `analyze_confirm.py` gives the same numbers when it resumes from
its checkpoints or is rerun on the same features. The G4 simulator embeds with fixed seeds. The stego images of the real
systems are not reproducible bit for bit: keys come from `keygen()` (OpenSSL) and messages from `secrets.token_bytes` at
run time, so a complete rerun reproduces the results statistically, not exactly. SRNet training on a GPU is not
bit-exact across hardware or library versions. The E4 run of the article used, as the message of every tenth session,
one of two fixed image files; the released `scripts/experiments/e4_functional.py` uses two synthetic binary messages of
the same lengths (919 and 2420 bytes, `pqpks.data.test_messages`), so a rerun of E4 gives results with the same
distribution, not the same bytes. Messages are encrypted before embedding, so only their lengths affect the embedding.
The warden scores of every cover, the extracted strings and the E1 syndromes of the run reported in the article are in
the dataset record; the SRM and SPAM feature matrices (about 58 GB of memmaps for E2/E3 alone) are not archived because
of their size.
