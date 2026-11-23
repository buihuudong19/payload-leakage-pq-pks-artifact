# run_all step table

Generated from `scripts/orchestration/pipeline.ps1` with `python scripts/orchestration/plan_table.py`. `$Py` is the
Python of `.venv-confirm`, `$PyG` the Python of `.venv-gpu`. A dependency `~X` only waits for step X to end; `file:P`
waits for file P; `serial:X` applies only when step G2b decides to run one GPU process at a time. Lanes W10 and W11 are
opened by hand (`run_lane.ps1 -Lane W10`, `-Lane W11`).

| # | Step | Lane | Command | Task | Waits for | Estimate | Output |
|---|---|---|---|---|---|---|---|
| 1 | 1 | W1 | `powershell -ExecutionPolicy Bypass -File scripts\confirm\setup_confirm.ps1` | Install .venv-confirm, check the profile, run the tests | – | 10-20 min | `results\confirm_setup` |
| 2 | 2 | W1 | `& $Py scripts\confirm\make_split.py --workers 7` | Split file split_v1 · checked after running | 1 | 10-50 min | `configs\splits\split_v1.json` |
| 3 | C1 | W1 | `& $Py scripts\confirm\srnet_prepare.py g4 --workers 7` | SRNet G4 data: 256x256, cover + HILL 0.4 | 2 | 15-30 min | `results\srnet\data_g4` |
| 4 | C2 | W1 | `& $Py scripts\confirm\e1_confirm.py extract --workers 7` | E1: extract Ext0 of the covers | C1 | 15-40 min | `results\e1_confirm` |
| 5 | C3 | W1 | `& $Py scripts\confirm\e1_confirm.py analyze` | E1: four distinguishers, G3, bit 3197 | C2 | 30-60 min | `results\e1_confirm\e1_summary.json` |
| 6 | C4 | W1 | `& $Py scripts\experiments\screen_run.py --plan confirm --srm --workers 6 --out results\confirm_v1\raw --save-stego all` | E2/E3 step 5a: 18 stego + SRM for 20 000 covers | C3 | 10.5-17.5 h | `results\confirm_v1\raw` |
| 7 | C6 | W1 | `& $Py scripts\confirm\analyze_confirm.py consolidate` | Collect features into memmaps | C4 | 30-90 min | `results\confirm_v1\feat` |
| 8 | C7 | W1 | `& $Py scripts\confirm\analyze_confirm.py analyze --reps 3` | SRM warden for 18 pairs: P_E, TOST, alpha*, H2a | C6, 3f | 4-20 h | `results\confirm_v1\summary.json` |
| 9 | C8 | W1 | `& $Py scripts\confirm\apply_rules.py --no-srnet` | Decision rules with SRM (+ sensitivity analysis) | C7 | < 1 min | `results\confirm_v1\decisions.json` |
| 10 | C9 | W1 | `& $Py scripts\experiments\e6_perf.py --img-reps 20 --out results\e6_desktop` | E6: send/receive timing (after C7, C8) | C8, ~E4, ~X5, ~C7, ~C8 | 15-45 min | `results\e6_desktop` |
| 11 | C10 | W1 | `& $Py scripts\experiments\e5_budget.py --name e5_summary_confirm.json` | E5 at the confirmatory alpha* (sigma by A5b; FN-DSA-512; exploratory A5c) | ~C9, ~C8 | < 1 min | `results\e5_budget\e5_summary_confirm.json` |
| 12 | R0 | W1 | `& $Py scripts\experiments\write_results_confirm.py` | Confirmatory tables (Tables 9-12, S1 -> RESULTS_TABLES.md) | ~C10, ~C3, ~X5, ~C8, ~C9, ~E4 | < 1 min | `results\RESULTS_TABLES.md` |
| 13 | R1 | W1 | `& $Py scripts\orchestration\collect_results.py` | Collect results | ~R0, ~E4, ~G13, ~G12, ~X5 | < 1 min | `results\manuscript_inputs` |
| 14 | 3a | W5 | `& $Py scripts\confirm\g4_calibrate.py features --workers 6` | SRM G4: HILL and S-UNIWARD features at 0.3 and 0.1 bpp | – | 3.5-4 h | `results\g4` |
| 15 | 3f | W5 | `& $Py scripts\confirm\g4_calibrate.py evaluate --seeds 3` | SRM G4: verdict (gate before the analysis) · checked after running | 3a, ~3e | 40 min | `results\g4\g4_summary.json` |
| 16 | 3e | W6 | `& $Py scripts\confirm\g4_audit_simulator.py --workers 4` | SRM G4: audit of the conseal simulator | file:results\g4\ids.json | 3-5 h | `results\g4\simulator_audit.json` |
| 17 | C5 | W6 | `& $Py scripts\confirm\srnet_prepare.py confirm --workers 6` | Confirmatory SRNet data 512x512 | ~3e, C4 | 30-60 min | `results\srnet\data_confirm` |
| 18 | X1 | W7 | `powershell -ExecutionPolicy Bypass -File scripts\data\download_alaska2.ps1 -List data\external\alaska2\e1bis_ids.txt` | E1-bis: download 40 000 ALASKA2 images (~10.5 GB) · checked after running | – | 2-8 h (network) | `data\external\raw\alaska2_tiff512` |
| 19 | X2 | W7 | `& $Py scripts\data\verify_alaska2.py --list data\external\alaska2\e1bis_ids.txt --budget 1e9` | E1-bis: verify images, convert to PGM, SHA-256 manifest | X1 | 20-60 min | `data\external\alaska2\pgm` |
| 20 | X3 | W7 | `& $Py scripts\confirm\e1bis.py split` | E1-bis: 5-fold split file (must equal the registered file) · checked after running | X2 | < 1 min | `configs\splits\e1bis_v1.json` |
| 21 | X4 | W7 | `& $Py scripts\confirm\e1bis.py extract --workers 4` | E1-bis: extract syndromes, 12 cells | X3 | 1-2 h | `results\e1bis\raw` |
| 22 | X5 | W7 | `& $Py scripts\confirm\e1bis.py analyze` | E1-bis: D1-D4, gate G3 of E1-bis | X4, file:results\e1bis_design\sd_null.json | 30-90 min | `results\e1bis\e1bis_summary.json` |
| 23 | D1 | W8 | `& $Py scripts\confirm\phase_d.py split-e7` | E7: 3-split file stratified by camera model (must equal the registered file) · checked after running | – | < 1 min | `configs\splits\e7_v1.json` |
| 24 | D2 | W8 | `& $Py scripts\experiments\screen_run.py --plan e7 --srm --workers 6 --out results\e7\raw` | E7: 8 stego + SRM for 10 000 ALASKA2 images (alpha = 0.2, 0.4) | D1 | 1.5-3 h | `results\e7\raw` |
| 25 | D3 | W8 | `& $Py scripts\confirm\analyze_confirm.py consolidate --base results\e7` | E7: collect features into memmaps | D2 | 10-30 min | `results\e7\feat` |
| 26 | D4 | W8 | `& $Py scripts\confirm\analyze_confirm.py analyze --base results\e7 --split e7 --cluster camera --reps 3` | E7: SRM warden, TOST, camera-cluster CI (8 pairs x 3 splits) | D3, ~C7, ~C9 | 2-4 h | `results\e7\summary.json` |
| 27 | D5 | W8 | `& $Py scripts\experiments\screen_run.py --plan ablate --srm --workers 6 --out results\ablate\raw` | S2: ablations A1 (V1-V3), A3 (S-UNIWARD), A4 (1252 B), A8p on 2000 screening covers | D4 | 1-3 h | `results\ablate\raw` |
| 28 | D6 | W8 | `& $Py scripts\confirm\analyze_confirm.py consolidate --base results\ablate` | S2: collect features | D5 | 5-15 min | `results\ablate\feat` |
| 29 | D7 | W8 | `& $Py scripts\confirm\analyze_confirm.py analyze --base results\ablate --split cv5 --reps 5 --tost-ref REF-U,PQ-PKS` | S2: SRM warden 5 folds, TOST against REF-U and PQ-PKS (18 pairs) | D6 | 3-6 h | `results\ablate\summary.json` |
| 30 | D8 | W8 | `& $Py scripts\experiments\screen_run.py --plan ra --srm --workers 6 --out results\ra\raw` | S5: regime R-a, the same 1024 and 4096 B messages for BL3, BL4, BL5, BL5-L, PQ-PKS, REF-U | D7 | 1-2 h | `results\ra\raw` |
| 31 | D9 | W8 | `& $Py scripts\confirm\analyze_confirm.py consolidate --base results\ra` | S5: collect features | D8 | 5-15 min | `results\ra\feat` |
| 32 | D10 | W8 | `& $Py scripts\confirm\analyze_confirm.py analyze --base results\ra --split cv5 --reps 5` | S5: SRM warden 5 folds (12 pairs) | D9 | 2-4 h | `results\ra\summary.json` |
| 33 | D11 | W8 | `& $Py scripts\confirm\phase_d.py tables` | Phase D: tables S2, S4, S5 | D10 | < 1 min | `results\phase_d\PHASE_D_TABLES_en.md` |
| 34 | D12 | W9 | `& $Py scripts\experiments\screen_run.py --plan ablate2 --srm --workers 6 --out results\ablate2\raw` | S2 part 2: A7 hybrid ML-KEM+X25519 (1568 B), A9 STC h = 8, 12 on 2000 screening covers · checked after running | ~D9 | 1-2 h | `results\ablate2\raw` |
| 35 | D13 | W9 | `& $Py scripts\confirm\analyze_confirm.py consolidate --base results\ablate2` | S2 part 2: collect features | D12 | 5-15 min | `results\ablate2\feat` |
| 36 | D14 | W9 | `& $Py scripts\confirm\analyze_confirm.py analyze --base results\ablate2 --split cv5 --reps 5 --tost-ref REF-U,PQ-PKS` | S2 part 2: SRM warden 5 folds, TOST against REF-U and PQ-PKS (12 pairs) | D13, ~C9 | 2-4 h | `results\ablate2\summary.json` |
| 37 | D15 | W9 | `& $Py scripts\experiments\jpeg_run.py --workers 6 --out results\jpeg\raw` | S3: JPEG QF75, DCT domain (J-UNIWARD), Q = 2048 and 3072 B, DCTR + GFR, 2000 screening covers | D14 | 2-4 h | `results\jpeg\raw` |
| 38 | D16 | W9 | `& $Py scripts\confirm\analyze_confirm.py consolidate --base results\jpeg` | S3: collect features | D15 | 5-15 min | `results\jpeg\feat` |
| 39 | D17 | W9 | `& $Py scripts\confirm\analyze_confirm.py analyze --base results\jpeg --split cv5 --reps 5 --wimg dctr` | S3: DCTR warden 5 folds, TOST (8 pairs) | D16 | 1-3 h | `results\jpeg\summary.json` |
| 40 | D18 | W9 | `& $Py scripts\confirm\analyze_confirm.py analyze --base results\jpeg --split cv5 --reps 5 --wimg gfr --out summary_gfr.json` | S3: GFR warden 5 folds, TOST (8 pairs) | D17 | 1-3 h | `results\jpeg\summary_gfr.json` |
| 41 | D19 | W9 | `& $Py scripts\confirm\phase_d.py tables` | Phase D: tables S2, S3, S4, S5 (update) | D18 | < 1 min | `results\phase_d\PHASE_D_TABLES_en.md` |
| 42 | PE1 | W10 | `& $Py scripts\confirm\analyze_confirm.py analyze --reps 10 --out summary_r10.json` | Package 1: C7 with 10 splits (Table 7: >= 10 splits) -> summary_r10.json | ~C7 | 40-75 h | `results\confirm_v1\summary_r10.json` |
| 43 | PE2 | W10 | `& $Py scripts\confirm\apply_rules.py --summary results\confirm_v1\summary_r10.json --out results\confirm_v1\decisions_r10.json --no-srnet` | Package 1: decision rules on the 10-split summary | PE1, PE1 | < 1 min | `results\confirm_v1\decisions_r10.json` |
| 44 | PE3 | W10 | `& $Py scripts\experiments\e6_perf.py --img-reps 20 --out results\e6_desktop` | Package 2: E6 remeasured (correct machine note; run on an idle CPU) | PE2, ~C9 | 15-45 min | `results\e6_desktop` |
| 45 | PE4 | W10 | `& $Py scripts\confirm\analyze_confirm.py analyze --base results\ra --split cv5 --reps 5` | Package 2: S5 reanalysed (D_pub of BL3 = point check; 12 pairs) | PE3, ~D10 | 2-4 h | `results\ra\summary.json` |
| 46 | PE5 | W10 | `& $Py scripts\experiments\screen_run.py --plan confirm_r --srm --workers 6 --out results\confirm_r\raw` | Package 3: PQ-PKS-R + REF-U-1252 at alpha = 0.04651, SRM, 20 000 covers | PE4, ~C4, ~PE4 | 1-2 h | `results\confirm_r\raw` |
| 47 | PE6 | W10 | `& $Py scripts\confirm\analyze_confirm.py consolidate --base results\confirm_r` | Package 3: collect features | PE5, PE5 | 10-30 min | `results\confirm_r\feat` |
| 48 | PE7 | W10 | `& $Py scripts\confirm\analyze_confirm.py analyze --base results\confirm_r --split confirm --reps 3 --tost-ref REF-U-1252` | Package 3: SRM warden, TOST of PQ-PKS-R against REF-U-1252 (2 pairs) | PE6, PE6 | 2-4 h | `results\confirm_r\summary.json` |
| 49 | PE8 | W10 | `& $Py scripts\confirm\variant_r.py` | Package 3: alpha*_R and the signature rule for variant R | PE7, PE7, ~C7 | < 1 min | `results\confirm_r\decisions_r.json` |
| 50 | PE9 | W10 | `& $Py scripts\experiments\jpeg_run.py --qf 95 --workers 6 --out results\jpeg95\raw` | Package 4: S3 JPEG QF95, Q = 2048 and 3072 B, DCTR + GFR, 2000 covers | PE8, ~PE8 | 2-4 h | `results\jpeg95\raw` |
| 51 | PE10 | W10 | `& $Py scripts\confirm\analyze_confirm.py consolidate --base results\jpeg95` | Package 4: collect features | PE9, PE9 | 5-15 min | `results\jpeg95\feat` |
| 52 | PE11 | W10 | `& $Py scripts\confirm\analyze_confirm.py analyze --base results\jpeg95 --split cv5 --reps 5 --wimg dctr` | Package 4: DCTR warden 5 folds, TOST (8 pairs) | PE10, PE10 | 1-3 h | `results\jpeg95\summary.json` |
| 53 | PE12 | W10 | `& $Py scripts\confirm\analyze_confirm.py analyze --base results\jpeg95 --split cv5 --reps 5 --wimg gfr --out summary_gfr.json` | Package 4: GFR warden 5 folds, TOST (8 pairs) | PE11, PE11 | 1-3 h | `results\jpeg95\summary_gfr.json` |
| 54 | PE13 | W10 | `& $Py scripts\confirm\phase_d.py tables` | Phases D/E: supplementary tables (update) | ~PE12 | < 1 min | `results\phase_d\PHASE_D_TABLES_en.md` |
| 55 | PE14 | W10 | `& $Py scripts\experiments\write_results_confirm.py` | Confirmatory tables (update after phase E) | ~PE13 | < 1 min | `results\RESULTS_TABLES.md` |
| 56 | PF1 | W11 | `& $Py scripts\confirm\analyze_confirm.py analyze --reps 10 --out summary_r10.json` | Final: C7 with 10 splits, bootstrap 10^4, H2a M = 4 (from checkpoints) | PE1 | 4-8 h | `results\confirm_v1\summary_r10.json` |
| 57 | PF2 | W11 | `& $Py scripts\confirm\promote_r10.py` | Final: the 10-split summary becomes summary.json (3-split version kept) | PF1, PF1, ~PE2 | < 1 min | `results\confirm_v1\summary.json` |
| 58 | PF3 | W11 | `& $Py scripts\confirm\apply_rules.py --no-srnet` | Final: decision rules (SRM) on the new main summary | PF2, PF2 | < 1 min | `results\confirm_v1\decisions.json` |
| 59 | PF4 | W11 | `& $Py scripts\experiments\e5_budget.py --name e5_summary_confirm.json --decisions results\srnet\decisions_srnet.json` | Final: E5 at the final alpha* (after SRNet; sigma by A5b) | PF3, PF3, G13 | < 1 min | `results\e5_budget\e5_summary_confirm.json` |
| 60 | PF5 | W11 | `& $Py scripts\confirm\phase_d.py tables` | Final: supplementary tables (phases D and E complete) | PF4, PF4, ~D19, ~PE4, ~PE8, ~PE12 | < 1 min | `results\phase_d\PHASE_D_TABLES_en.md` |
| 61 | PF6 | W11 | `& $Py scripts\experiments\write_results_confirm.py` | Final: Tables 9-12, S1 (RESULTS_TABLES.md) | PF5, PF5, ~G12 | < 1 min | `results\RESULTS_TABLES.md` |
| 62 | PF7 | W11 | `& $Py scripts\orchestration\collect_results.py` | Final: collect results | PF6, PF6, ~R1 | < 1 min | `results\manuscript_inputs` |
| 63 | PF8 | W11 | `& $Py scripts\figures\make_all_figures.py` | Final: regenerate every figure (results\figures) | PF7, PF7 | 2-5 min | `results\figures_confirm` |
| 64 | G1 | W2 | `powershell -ExecutionPolicy Bypass -File scripts\gpu\setup_gpu.ps1` | Install .venv-gpu (torch CUDA) · checked after running | – | 10-30 min | `.venv-gpu` |
| 65 | G3a | W2 | `& $PyG scripts\gpu\srnet_train.py --run g4_hill_0.4 --until 200000` | SRNet G4 HILL 0.4: 200k iterations lr 1e-3 (branch point) | G1, C1 | ~11 h | `results\srnet\g4_hill_0.4` |
| 66 | G3b | W2 | `& $PyG scripts\gpu\srnet_train.py --run g4_hill_0.4_short` | SRNet G4 short branch: 50k iterations lr 1e-4 | G3a | ~3 h | `results\srnet\g4_hill_0.4_short` |
| 67 | G7a | W2 | `& $Py scripts\confirm\srnet_evaluate.py g4` | SRNet G4 verdict (short branch) · checked after running | G3b | < 5 min | `results\srnet\g4_srnet_summary.json` |
| 68 | G3c | W2 | `& $PyG scripts\gpu\srnet_train.py --run g4_hill_0.4` | Fallback: G4 HILL 0.4 full 500k schedule (only if the short branch fails) (skipped when: the short branch passed G4: no full schedule needed) | G7a | ~17 h | `results\srnet\g4_hill_0.4` |
| 69 | G7b | W2 | `& $Py scripts\confirm\srnet_evaluate.py g4` | SRNet G4 verdict (full schedule) (skipped when: full schedule not run) · checked after running | G3c | < 5 min | `results\srnet\g4_srnet_summary.json` |
| 70 | G2 | W2 | `& $PyG scripts\gpu\bench_srnet.py --quick` | GPU speed benchmark (idle GPU, before confirmatory SRNet) | G7b | 3-5 min | `results\gpu_bench` |
| 71 | G2b | W2 | `& $Py scripts\orchestration\gpu_decide.py` | Decide on 1 or 2 GPU processes | ~G2 | < 1 min | `results\gpu_bench\decision.json` |
| 72 | G8 | W2 | `& $PyG scripts\gpu\srnet_train.py --run confirm_REF-U_0.4` | Confirmatory SRNet REF-U 0.4 (150k iterations) | ~G2b, G7a, G7b, C5 | ~8 h (~16 in parallel) | `results\srnet\confirm_REF-U_0.4` |
| 73 | G9 | W2 | `& $PyG scripts\gpu\srnet_train.py --run confirm_REF-U_0.2` | Confirmatory SRNet REF-U 0.2 (100k iterations) | G8 | ~6 h (~11 in parallel) | `results\srnet\confirm_REF-U_0.2` |
| 74 | G12 | W2 | `& $Py scripts\confirm\srnet_evaluate.py confirm` | Confirmatory SRNet: P_E, TOST, McNemar | G9, G11 | < 5 min | `results\srnet\confirm_srnet_summary.json` |
| 75 | G13 | W2 | `& $Py scripts\confirm\apply_rules.py --out results\srnet\decisions_srnet.json` | Decision rules with SRNet included | G12, C8 | < 1 min | `results\srnet\decisions_srnet.json` |
| 76 | G10 | W3 | `& $PyG scripts\gpu\srnet_train.py --run confirm_PQ-PKS_0.4` | Confirmatory SRNet PQ-PKS 0.4 (150k iterations) | G7a, G7b, C5, serial:G9 | ~8 h (~16 in parallel) | `results\srnet\confirm_PQ-PKS_0.4` |
| 77 | G11 | W3 | `& $PyG scripts\gpu\srnet_train.py --run confirm_PQ-PKS_0.2` | Confirmatory SRNet PQ-PKS 0.2 (100k iterations) | G10 | ~6 h (~11 in parallel) | `results\srnet\confirm_PQ-PKS_0.2` |
| 78 | E4 | W4 | `& $Py scripts\experiments\e4_functional.py --size 512 --normal 100000 --tamper 10000 --cover-images 900 --out results\e4_full` | Full-scale E4 on 512x512 images | 1, 2 | 25-35 h | `results\e4_full\e4_summary.json` |
