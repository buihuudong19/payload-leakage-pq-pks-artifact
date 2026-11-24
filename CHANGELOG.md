# Changelog

Nothing in `src/` or `configs/` has changed since the frozen profile of 5 November 2026; every release adds scripts,
tests and documentation around the frozen code.

## v1.0.2 (24 November 2026)

Steps registered after the review of 24 November 2026 (lane W13, steps RV1-RV10; see section 9 of
`docs/reproduction.md`). Their plans were committed and pushed before any of their data existed (commit `d32c1725`).

- `registrations/`: plans of G4 on ten fresh seeds (`g4_new_seeds_plan.json`) and of the second replication E1-ter
  (`e1ter_plan.json`, `E1TER_PLAN.md`); `data/external/alaska2/e1ter_ids.txt` (38 533 ALASKA2 images that no earlier
  step used, fixed by its SHA-256).
- `scripts/confirm/g4_new_seeds.py` (RV1), `scripts/data/make_alaska2_e1ter_subset.py` (RV2) and
  `scripts/confirm/e1ter.py` (RV3, RV4, RV7, RV8: split, null draws with retraining, extraction, analysis).
- Post hoc: `scripts/experiments/receiver_fast.py` (receiver that caches the session-keyed stage-2 permutations, same
  outputs as the reference receiver) and `scripts/experiments/e6_load.py` (latency percentiles and throughput of invalid
  images under concurrent load, RV9); `scripts/theory/epsilon_valid_rom.py` (bound on epsilon_valid with G as a random
  oracle, RV10).
- `scripts/verify/clean_env_check.ps1` / `.py` / `.cmd`: checks a release in a new folder and virtual environment against
  the dataset record.
- `scripts/setup/check_bl1_reference.py` and `baselines/bl1_reference.SHA256SUMS`: hashes of the BL1 reference modules,
  which are not redistributed.
- `scripts/orchestration/run_review_steps.ps1` / `.cmd` and lane W13 in `pipeline.ps1` and `run_lane.ps1`.
- Fixes found while running W13 on 24 November 2026: `run_review_steps.ps1` runs Python in UTF-8 mode (RV6 stopped on a
  console-encoding error after converting every image); `e6_load.py` refuses concurrency levels above the number of
  logical processors (64 processes exhausted the memory of a 32 GB desktop), stops with an error when a worker dies
  instead of waiting at the barrier, and writes its summary after every level so that a stopped run can be continued.
- Tests: `test_g4_new_seeds.py`, `test_e1ter.py`, `test_receiver_fast.py`, `test_epsilon_valid_rom.py`,
  `test_check_bl1_reference.py`; `test_orchestration.py` covers lane W13. 238 tests in total.

## v1.0.1 (24 November 2026)

- Post hoc lane W12: `scripts/confirm/e1_posthoc_refit.py` (bootstrap of the learned distinguisher D4 of E1 that refits
  the models, with its null calibration) and its tests.
- Co-author added to `CITATION.cff`; DOI of the dataset record version 1.1.

## v1.0.0 (23 November 2026)

First release: the code of every experiment of the article.
