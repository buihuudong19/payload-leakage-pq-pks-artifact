# Step table of run_all (dot-sourced). A single table: run_lane.ps1 runs the steps of one lane in order and
# monitor.ps1 shows their status.
#
# Lanes: W1 main CPU, W2 GPU (SRNet G4, GPU benchmark while the GPU is idle, then the REF-U chain), W3 GPU (PQ-PKS chain; runs in
# parallel with W2 only if step G2b decides so), W4 single-threaded E4, W5 and W6 secondary CPU lanes.
# Data dependencies: 3a reads only BOSSbase; 3e also reads results\g4\ids.json, which 3a writes when it starts (dep file:); C5 reads
# the stego images of C4; 3f reads the features of 3a and simulator_audit.json of 3e; C7 reads g4_summary.json of 3f (sensitivity analysis).
# Each step implicitly depends on the step before it in the same lane (Soft = only wait for the previous step to end).
# Kind: py = .venv-confirm, pyg = .venv-gpu, ps = powershell -File.
# When: run condition; if false the step is marked skipped. Check: run after the step; a returned string marks the step failed.

$LaneTitle = [ordered]@{
    W1 = 'CPU: splits, E1, E2/E3, SRM G4, analysis'
    W2 = 'GPU 1: SRNet G4, confirmatory REF-U'
    W3 = 'GPU 2: confirmatory PQ-PKS'
    W4 = 'CPU, 1 thread: full-scale E4 512x512'
    W5 = 'CPU 2: SRM G4 (features, verdict)'
    W6 = 'CPU 3: G4 simulator audit, confirmatory SRNet data'
    W7 = 'E1-bis: ALASKA2 download, syndrome extraction, gate G3 of E1-bis'
    W8 = 'Phase D: E7 ALASKA2, ablations S2, regime R-a S5'
    W9 = 'Phase D part 2: A7 hybrid, A9 STC h = 8/12, S3 JPEG QF75'
}
# Manual lanes: not opened by run_all; each is started separately with run_lane.ps1 -Lane <name>.
# Their steps are still shown by the monitor (state "pending" until the lane is opened).
$ManualLanes = [ordered]@{
    W10 = 'Phase E (manual): C7 with 10 splits, C9 remeasured, S5 with D_pub of BL3, variant R at 0.04651, JPEG QF95'
    W11 = 'Final aggregation: bootstrap 10^4 + M = 4, 10-split summary as main summary, E5/tables/figures after SRNet'
    W12 = 'Post-hoc analyses (not registered): E1 D4 null check and model-refit bootstrap, alpha* threshold sensitivity'
}
$LaneTitleAll = [ordered]@{}
foreach ($k in $LaneTitle.Keys) { $LaneTitleAll[$k] = $LaneTitle[$k] }
foreach ($k in $ManualLanes.Keys) { $LaneTitleAll[$k] = $ManualLanes[$k] }

$SplitSha256 = '8d38c3aab7a773758ecb755f208accdc39b5b2db87da21eab1db5554f1001b57'   # split_v1, generated on 06/11/2026
$E1bisIdsSha256 = '88cbe2e85bae9e930bcb412a422779e50598f4230f1f5631c9e9aa289f3b1f5a'    # e1bis_ids.txt
$E1bisSplitSha256 = '18e1c90da36f74c2b330e039cb04a284c662ea6be89083dda6fbd6d2f742daca'  # e1bis_v1.json
$E7SplitSha256 = 'cbc62449b63eb82ac129adc70f7387d8d76225dea122e453cdff97cfeb4f0188'     # e7_v1.json (generated on 08/11/2026, before any stego image)

$Steps = @(
    # ---------------- W1 ----------------
    @{ Id = '1'; Lane = 'W1'; Name = 'Install .venv-confirm, check the profile, run the tests'; Kind = 'ps'
       Args = @('scripts\confirm\setup_confirm.ps1'); Deps = @(); Est = '10-20 min'; Out = 'results\confirm_setup' }
    @{ Id = '2'; Lane = 'W1'; Name = 'Split file split_v1'; Kind = 'py'
       Args = @('scripts\confirm\make_split.py', '--workers', '7'); Deps = @(); Est = '10-50 min'; Out = 'configs\splits\split_v1.json'
       Check = {   # the split file must equal the registered file byte for byte (SHA-256 8d38c3aa...)
           $f = Join-Path $Root 'configs\splits\split_v1.json'
           if (-not (Test-Path $f)) { 'configs\splits\split_v1.json not found' }
           elseif ((Get-FileHash -Algorithm SHA256 $f).Hash.ToLower() -ne $SplitSha256) { "split_v1.json differs from the registered file (SHA-256 $SplitSha256)" } } }
    @{ Id = 'C1'; Lane = 'W1'; Name = 'SRNet G4 data: 256x256, cover + HILL 0.4'; Kind = 'py'
       Args = @('scripts\confirm\srnet_prepare.py', 'g4', '--workers', '7'); Deps = @(); Est = '15-30 min'; Out = 'results\srnet\data_g4' }
    @{ Id = 'C2'; Lane = 'W1'; Name = 'E1: extract Ext0 of the covers'; Kind = 'py'
       Args = @('scripts\confirm\e1_confirm.py', 'extract', '--workers', '7'); Deps = @(); Est = '15-40 min'; Out = 'results\e1_confirm' }
    @{ Id = 'C3'; Lane = 'W1'; Name = 'E1: four distinguishers, G3, bit 3197'; Kind = 'py'
       Args = @('scripts\confirm\e1_confirm.py', 'analyze'); Deps = @(); Est = '30-60 min'; Out = 'results\e1_confirm\e1_summary.json' }
    @{ Id = 'C4'; Lane = 'W1'; Name = 'E2/E3 step 5a: 18 stego + SRM for 20 000 covers'; Kind = 'py'
       Args = @('scripts\experiments\screen_run.py', '--plan', 'confirm', '--srm', '--workers', '6', '--out', 'results\confirm_v1\raw', '--save-stego', 'all')
       Deps = @(); Est = '10.5-17.5 h'; Out = 'results\confirm_v1\raw' }
    @{ Id = 'C6'; Lane = 'W1'; Name = 'Collect features into memmaps'; Kind = 'py'
       Args = @('scripts\confirm\analyze_confirm.py', 'consolidate'); Deps = @(); Est = '30-90 min'; Out = 'results\confirm_v1\feat' }
    @{ Id = 'C7'; Lane = 'W1'; Name = 'SRM warden for 18 pairs: P_E, TOST, alpha*, H2a'; Kind = 'py'
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--reps', '3'); Deps = @('3f'); Est = '4-20 h'; Out = 'results\confirm_v1\summary.json' }
    @{ Id = 'C8'; Lane = 'W1'; Name = 'Decision rules with SRM (+ sensitivity analysis)'; Kind = 'py'
       Args = @('scripts\confirm\apply_rules.py', '--no-srnet'); Deps = @(); Est = '< 1 min'; Out = 'results\confirm_v1\decisions.json' }
    # C9 does not wait for G13 (its only real inputs are E4 and X5); it runs right after C7 and C8.
    @{ Id = 'C9'; Lane = 'W1'; Name = 'E6: send/receive timing (after C7, C8)'; Kind = 'py'
       Args = @('scripts\experiments\e6_perf.py', '--img-reps', '20', '--out', 'results\e6_desktop'); Deps = @('~E4', '~X5', '~C7', '~C8')
       Est = '15-45 min'; Out = 'results\e6_desktop' }
    # E5 recomputed at the confirmatory alpha* with the placement of sigma given by the A5b rule (decisions.json), then the
    # confirmatory RESULTS_TABLES.md (Tables 9-12, S1) is written.
    @{ Id = 'C10'; Lane = 'W1'; Name = 'E5 at the confirmatory alpha* (sigma by A5b; FN-DSA-512; exploratory A5c)'; Kind = 'py'; Soft = $true
       Args = @('scripts\experiments\e5_budget.py', '--name', 'e5_summary_confirm.json'); Deps = @('~C8'); Est = '< 1 min'
       Out = 'results\e5_budget\e5_summary_confirm.json' }
    @{ Id = 'R0'; Lane = 'W1'; Name = 'Confirmatory tables (Tables 9-12, S1 -> RESULTS_TABLES.md)'; Kind = 'py'; Soft = $true
       Args = @('scripts\experiments\write_results_confirm.py'); Deps = @('~C3', '~X5', '~C8', '~C9', '~E4'); Est = '< 1 min'
       Out = 'results\RESULTS_TABLES.md' }
    @{ Id = 'R1'; Lane = 'W1'; Name = 'Collect results'; Kind = 'py'; Soft = $true
       Args = @('scripts\orchestration\collect_results.py'); Deps = @('~E4', '~G13', '~G12', '~X5'); Est = '< 1 min'; Out = 'results\manuscript_inputs' }

    # ---------------- W5 ----------------
    @{ Id = '3a'; Lane = 'W5'; Name = 'SRM G4: HILL and S-UNIWARD features at 0.3 and 0.1 bpp'; Kind = 'py'
       Args = @('scripts\confirm\g4_calibrate.py', 'features', '--workers', '6'); Deps = @(); Est = '3.5-4 h'; Out = 'results\g4' }
    @{ Id = '3f'; Lane = 'W5'; Name = 'SRM G4: verdict (gate before the analysis)'; Kind = 'py'
       Args = @('scripts\confirm\g4_calibrate.py', 'evaluate', '--seeds', '3'); Deps = @('~3e'); Est = '40 min'; Out = 'results\g4\g4_summary.json'
       Check = { $g = Read-JsonFile 'results\g4\g4_summary.json'; if (-not $g -or -not $g.G4_pass) { 'SRM G4 failed: stopping before the analysis (see g4_summary.json)' } } }

    # ---------------- W6 ----------------
    @{ Id = '3e'; Lane = 'W6'; Name = 'SRM G4: audit of the conseal simulator'; Kind = 'py'
       Args = @('scripts\confirm\g4_audit_simulator.py', '--workers', '4'); Deps = @('file:results\g4\ids.json'); Est = '3-5 h'; Out = 'results\g4\simulator_audit.json' }
    @{ Id = 'C5'; Lane = 'W6'; Name = 'Confirmatory SRNet data 512x512'; Kind = 'py'; Soft = $true
       Args = @('scripts\confirm\srnet_prepare.py', 'confirm', '--workers', '6'); Deps = @('C4'); Est = '30-60 min'; Out = 'results\srnet\data_confirm' }

    # ---------------- W7: E1-bis ----------------
    @{ Id = 'X1'; Lane = 'W7'; Name = 'E1-bis: download 40 000 ALASKA2 images (~10.5 GB)'; Kind = 'ps'
       Args = @('scripts\data\download_alaska2.ps1', '-List', 'data\external\alaska2\e1bis_ids.txt'); Deps = @(); Est = '2-8 h (network)'
       Out = 'data\external\raw\alaska2_tiff512'
       Check = { $f = Join-Path $Root 'data\external\alaska2\e1bis_ids.txt'
                 if ((Get-FileHash -Algorithm SHA256 $f).Hash.ToLower() -ne $E1bisIdsSha256) { "e1bis_ids.txt differs from the registered file ($E1bisIdsSha256)" } } }
    @{ Id = 'X2'; Lane = 'W7'; Name = 'E1-bis: verify images, convert to PGM, SHA-256 manifest'; Kind = 'py'
       Args = @('scripts\data\verify_alaska2.py', '--list', 'data\external\alaska2\e1bis_ids.txt', '--budget', '1e9'); Deps = @(); Est = '20-60 min'
       Out = 'data\external\alaska2\pgm' }
    @{ Id = 'X3'; Lane = 'W7'; Name = 'E1-bis: 5-fold split file (must equal the registered file)'; Kind = 'py'
       Args = @('scripts\confirm\e1bis.py', 'split'); Deps = @(); Est = '< 1 min'; Out = 'configs\splits\e1bis_v1.json'
       Check = { $f = Join-Path $Root 'configs\splits\e1bis_v1.json'
                 if ((Get-FileHash -Algorithm SHA256 $f).Hash.ToLower() -ne $E1bisSplitSha256) { "e1bis_v1.json differs from the registered file ($E1bisSplitSha256)" } } }
    @{ Id = 'X4'; Lane = 'W7'; Name = 'E1-bis: extract syndromes, 12 cells'; Kind = 'py'
       Args = @('scripts\confirm\e1bis.py', 'extract', '--workers', '4'); Deps = @(); Est = '1-2 h'; Out = 'results\e1bis\raw' }
    @{ Id = 'X5'; Lane = 'W7'; Name = 'E1-bis: D1-D4, gate G3 of E1-bis'; Kind = 'py'
       Args = @('scripts\confirm\e1bis.py', 'analyze'); Deps = @('file:results\e1bis_design\sd_null.json'); Est = '30-90 min'
       Out = 'results\e1bis\e1bis_summary.json' }

    # ---------------- W8: phase D.
    # The stego generation and collection steps (D1-D3, D5-D6, D8-D9, D12-D13, D15-D16) do not depend on C7 and can run in parallel
    # with it; the analysis steps (D4, D14) wait for ~C9 so that they do not disturb the timing measurement.
    # Cells: number of (cell, system) pairs trained by analyze_confirm (used by the progress estimate of progress.ps1). ----------------
    @{ Id = 'D1'; Lane = 'W8'; Name = 'E7: 3-split file stratified by camera model (must equal the registered file)'; Kind = 'py'
       Args = @('scripts\confirm\phase_d.py', 'split-e7'); Deps = @(); Est = '< 1 min'; Out = 'configs\splits\e7_v1.json'
       Check = { $f = Join-Path $Root 'configs\splits\e7_v1.json'
                 if ((Get-FileHash -Algorithm SHA256 $f).Hash.ToLower() -ne $E7SplitSha256) { "e7_v1.json differs from the registered file ($E7SplitSha256)" } } }
    @{ Id = 'D2'; Lane = 'W8'; Name = 'E7: 8 stego + SRM for 10 000 ALASKA2 images (alpha = 0.2, 0.4)'; Kind = 'py'
       Args = @('scripts\experiments\screen_run.py', '--plan', 'e7', '--srm', '--workers', '6', '--out', 'results\e7\raw'); Deps = @()
       Est = '1.5-3 h'; Out = 'results\e7\raw' }
    @{ Id = 'D3'; Lane = 'W8'; Name = 'E7: collect features into memmaps'; Kind = 'py'
       Args = @('scripts\confirm\analyze_confirm.py', 'consolidate', '--base', 'results\e7'); Deps = @(); Est = '10-30 min'; Out = 'results\e7\feat' }
    @{ Id = 'D4'; Lane = 'W8'; Name = 'E7: SRM warden, TOST, camera-cluster CI (8 pairs x 3 splits)'; Kind = 'py'; Cells = 8
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--base', 'results\e7', '--split', 'e7', '--cluster', 'camera', '--reps', '3')
       Deps = @('~C7', '~C9'); Est = '2-4 h'; Out = 'results\e7\summary.json' }
    @{ Id = 'D5'; Lane = 'W8'; Name = 'S2: ablations A1 (V1-V3), A3 (S-UNIWARD), A4 (1252 B), A8p on 2000 screening covers'; Kind = 'py'
       Args = @('scripts\experiments\screen_run.py', '--plan', 'ablate', '--srm', '--workers', '6', '--out', 'results\ablate\raw'); Deps = @()
       Est = '1-3 h'; Out = 'results\ablate\raw' }
    @{ Id = 'D6'; Lane = 'W8'; Name = 'S2: collect features'; Kind = 'py'
       Args = @('scripts\confirm\analyze_confirm.py', 'consolidate', '--base', 'results\ablate'); Deps = @(); Est = '5-15 min'; Out = 'results\ablate\feat' }
    @{ Id = 'D7'; Lane = 'W8'; Name = 'S2: SRM warden 5 folds, TOST against REF-U and PQ-PKS (18 pairs)'; Kind = 'py'; Cells = 18
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--base', 'results\ablate', '--split', 'cv5', '--reps', '5', '--tost-ref', 'REF-U,PQ-PKS')
       Deps = @(); Est = '3-6 h'; Out = 'results\ablate\summary.json' }
    @{ Id = 'D8'; Lane = 'W8'; Name = 'S5: regime R-a, the same 1024 and 4096 B messages for BL3, BL4, BL5, BL5-L, PQ-PKS, REF-U'; Kind = 'py'
       Args = @('scripts\experiments\screen_run.py', '--plan', 'ra', '--srm', '--workers', '6', '--out', 'results\ra\raw'); Deps = @()
       Est = '1-2 h'; Out = 'results\ra\raw' }
    @{ Id = 'D9'; Lane = 'W8'; Name = 'S5: collect features'; Kind = 'py'
       Args = @('scripts\confirm\analyze_confirm.py', 'consolidate', '--base', 'results\ra'); Deps = @(); Est = '5-15 min'; Out = 'results\ra\feat' }
    @{ Id = 'D10'; Lane = 'W8'; Name = 'S5: SRM warden 5 folds (12 pairs)'; Kind = 'py'; Cells = 12
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--base', 'results\ra', '--split', 'cv5', '--reps', '5')
       Deps = @(); Est = '2-4 h'; Out = 'results\ra\summary.json' }
    @{ Id = 'D11'; Lane = 'W8'; Name = 'Phase D: tables S2, S4, S5'; Kind = 'py'
       Args = @('scripts\confirm\phase_d.py', 'tables'); Deps = @(); Est = '< 1 min'; Out = 'results\phase_d\PHASE_D_TABLES_en.md' }

    # ---------------- W9: phase D part 2. Runs after lane W8 (D11) to avoid competing for the CPU; D12 waits for ~D9.
    # The H_hat tables for h = 8, 12 are built deterministically (scripts/experiments/stc_tables/). ----------------
    @{ Id = 'D12'; Lane = 'W9'; Name = 'S2 part 2: A7 hybrid ML-KEM+X25519 (1568 B), A9 STC h = 8, 12 on 2000 screening covers'; Kind = 'py'
       Args = @('scripts\experiments\screen_run.py', '--plan', 'ablate2', '--srm', '--workers', '6', '--out', 'results\ablate2\raw'); Deps = @('~D9')
       Est = '1-2 h'; Out = 'results\ablate2\raw'
       Check = { foreach ($h in 8, 12) { if (-not (Test-Path (Join-Path $Root "scripts\experiments\stc_tables\stc_table_h${h}.json"))) { "H_hat table missing for h = $h" } } } }
    @{ Id = 'D13'; Lane = 'W9'; Name = 'S2 part 2: collect features'; Kind = 'py'
       Args = @('scripts\confirm\analyze_confirm.py', 'consolidate', '--base', 'results\ablate2'); Deps = @(); Est = '5-15 min'; Out = 'results\ablate2\feat' }
    @{ Id = 'D14'; Lane = 'W9'; Name = 'S2 part 2: SRM warden 5 folds, TOST against REF-U and PQ-PKS (12 pairs)'; Kind = 'py'; Cells = 12
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--base', 'results\ablate2', '--split', 'cv5', '--reps', '5', '--tost-ref', 'REF-U,PQ-PKS')
       Deps = @('~C9'); Est = '2-4 h'; Out = 'results\ablate2\summary.json' }
    @{ Id = 'D15'; Lane = 'W9'; Name = 'S3: JPEG QF75, DCT domain (J-UNIWARD), Q = 2048 and 3072 B, DCTR + GFR, 2000 screening covers'; Kind = 'py'
       Args = @('scripts\experiments\jpeg_run.py', '--workers', '6', '--out', 'results\jpeg\raw'); Deps = @(); Est = '2-4 h'; Out = 'results\jpeg\raw' }
    @{ Id = 'D16'; Lane = 'W9'; Name = 'S3: collect features'; Kind = 'py'
       Args = @('scripts\confirm\analyze_confirm.py', 'consolidate', '--base', 'results\jpeg'); Deps = @(); Est = '5-15 min'; Out = 'results\jpeg\feat' }
    @{ Id = 'D17'; Lane = 'W9'; Name = 'S3: DCTR warden 5 folds, TOST (8 pairs)'; Kind = 'py'; Cells = 8
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--base', 'results\jpeg', '--split', 'cv5', '--reps', '5', '--wimg', 'dctr')
       Deps = @(); Est = '1-3 h'; Out = 'results\jpeg\summary.json' }
    @{ Id = 'D18'; Lane = 'W9'; Name = 'S3: GFR warden 5 folds, TOST (8 pairs)'; Kind = 'py'; Cells = 8
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--base', 'results\jpeg', '--split', 'cv5', '--reps', '5', '--wimg', 'gfr', '--out', 'summary_gfr.json')
       Deps = @(); Est = '1-3 h'; Out = 'results\jpeg\summary_gfr.json' }
    @{ Id = 'D19'; Lane = 'W9'; Name = 'Phase D: tables S2, S3, S4, S5 (update)'; Kind = 'py'
       Args = @('scripts\confirm\phase_d.py', 'tables'); Deps = @(); Est = '< 1 min'; Out = 'results\phase_d\PHASE_D_TABLES_en.md' }

    # ---------------- W10: phase E, manual lane ----------------
    # Package 1 (PE1, PE2): C7 with 10 splits as registered for Table 7 (C7 ran with 3). Writes summary_r10.json; the 3-split summary is not overwritten.
    @{ Id = 'PE1'; Lane = 'W10'; Name = 'Package 1: C7 with 10 splits (Table 7: >= 10 splits) -> summary_r10.json'; Kind = 'py'; Cells = 18
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--reps', '10', '--out', 'summary_r10.json'); Deps = @('~C7')
       Est = '40-75 h'; Out = 'results\confirm_v1\summary_r10.json' }
    @{ Id = 'PE2'; Lane = 'W10'; Name = 'Package 1: decision rules on the 10-split summary'; Kind = 'py'
       Args = @('scripts\confirm\apply_rules.py', '--summary', 'results\confirm_v1\summary_r10.json', '--out', 'results\confirm_v1\decisions_r10.json', '--no-srnet')
       Deps = @('PE1'); Est = '< 1 min'; Out = 'results\confirm_v1\decisions_r10.json' }
    # Package 2 (PE3, PE4): C9 remeasured (the earlier version is kept under another name) and S5 reanalysed with D_pub of BL3 = P-256 point check.
    @{ Id = 'PE3'; Lane = 'W10'; Name = 'Package 2: E6 remeasured (correct machine note; run on an idle CPU)'; Kind = 'py'
       Args = @('scripts\experiments\e6_perf.py', '--img-reps', '20', '--out', 'results\e6_desktop'); Deps = @('~C9'); Est = '15-45 min'
       Out = 'results\e6_desktop' }
    @{ Id = 'PE4'; Lane = 'W10'; Name = 'Package 2: S5 reanalysed (D_pub of BL3 = point check; 12 pairs)'; Kind = 'py'; Cells = 12
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--base', 'results\ra', '--split', 'cv5', '--reps', '5'); Deps = @('~D10')
       Est = '2-4 h'; Out = 'results\ra\summary.json' }
    # Order: packages 1 and 2 first; package 3 after package 2 (PE5 waits for ~PE4); package 4 after package 3 (PE9 waits for ~PE8).
    # Package 3 (PE5-PE8): rejection-sampled Kemeleon variant (A4, 1252 B) at its smallest session-start image 0.04651 on the 20 000 confirmatory covers (exploratory).
    @{ Id = 'PE5'; Lane = 'W10'; Name = 'Package 3: PQ-PKS-R + REF-U-1252 at alpha = 0.04651, SRM, 20 000 covers'; Kind = 'py'
       Args = @('scripts\experiments\screen_run.py', '--plan', 'confirm_r', '--srm', '--workers', '6', '--out', 'results\confirm_r\raw'); Deps = @('~C4', '~PE4')
       Est = '1-2 h'; Out = 'results\confirm_r\raw' }
    @{ Id = 'PE6'; Lane = 'W10'; Name = 'Package 3: collect features'; Kind = 'py'
       Args = @('scripts\confirm\analyze_confirm.py', 'consolidate', '--base', 'results\confirm_r'); Deps = @('PE5'); Est = '10-30 min'; Out = 'results\confirm_r\feat' }
    @{ Id = 'PE7'; Lane = 'W10'; Name = 'Package 3: SRM warden, TOST of PQ-PKS-R against REF-U-1252 (2 pairs)'; Kind = 'py'; Cells = 2
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--base', 'results\confirm_r', '--split', 'confirm', '--reps', '3', '--tost-ref', 'REF-U-1252')
       Deps = @('PE6'); Est = '2-4 h'; Out = 'results\confirm_r\summary.json' }
    @{ Id = 'PE8'; Lane = 'W10'; Name = 'Package 3: alpha*_R and the signature rule for variant R'; Kind = 'py'
       Args = @('scripts\confirm\variant_r.py'); Deps = @('PE7', '~C7'); Est = '< 1 min'; Out = 'results\confirm_r\decisions_r.json' }
    # Package 4 (PE9-PE12): S3 at JPEG QF95 (larger |U|, P_E not at the floor) on the 2000 screening covers.
    @{ Id = 'PE9'; Lane = 'W10'; Name = 'Package 4: S3 JPEG QF95, Q = 2048 and 3072 B, DCTR + GFR, 2000 covers'; Kind = 'py'
       Args = @('scripts\experiments\jpeg_run.py', '--qf', '95', '--workers', '6', '--out', 'results\jpeg95\raw'); Deps = @('~PE8'); Est = '2-4 h'; Out = 'results\jpeg95\raw' }
    @{ Id = 'PE10'; Lane = 'W10'; Name = 'Package 4: collect features'; Kind = 'py'
       Args = @('scripts\confirm\analyze_confirm.py', 'consolidate', '--base', 'results\jpeg95'); Deps = @('PE9'); Est = '5-15 min'; Out = 'results\jpeg95\feat' }
    @{ Id = 'PE11'; Lane = 'W10'; Name = 'Package 4: DCTR warden 5 folds, TOST (8 pairs)'; Kind = 'py'; Cells = 8
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--base', 'results\jpeg95', '--split', 'cv5', '--reps', '5', '--wimg', 'dctr')
       Deps = @('PE10'); Est = '1-3 h'; Out = 'results\jpeg95\summary.json' }
    @{ Id = 'PE12'; Lane = 'W10'; Name = 'Package 4: GFR warden 5 folds, TOST (8 pairs)'; Kind = 'py'; Cells = 8
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--base', 'results\jpeg95', '--split', 'cv5', '--reps', '5', '--wimg', 'gfr', '--out', 'summary_gfr.json')
       Deps = @('PE11'); Est = '1-3 h'; Out = 'results\jpeg95\summary_gfr.json' }
    # End of phase E: supplementary tables and confirmatory tables; may run after any package (Soft: waits only for the previous step in the window).
    @{ Id = 'PE13'; Lane = 'W10'; Name = 'Phases D/E: supplementary tables (update)'; Kind = 'py'; Soft = $true
       Args = @('scripts\confirm\phase_d.py', 'tables'); Deps = @(); Est = '< 1 min'; Out = 'results\phase_d\PHASE_D_TABLES_en.md' }
    @{ Id = 'PE14'; Lane = 'W10'; Name = 'Confirmatory tables (update after phase E)'; Kind = 'py'; Soft = $true
       Args = @('scripts\experiments\write_results_confirm.py'); Deps = @(); Est = '< 1 min'; Out = 'results\RESULTS_TABLES.md' }

    # ---------------- W11: final aggregation, manual lane ----------------
    # The window waits for PE1 and then for G13 (W2), so it can be opened at once.
    # PF1 reruns the 10-split analysis from the _R10 checkpoints (no retraining) with every bootstrap at 10^4 resamples and H2a
    # with M = 4; PF2 makes the 10-split summary summary.json (the 3-split version is kept as summary_r3.json); PF3 recomputes
    # decisions.json. PF1-PF3 must finish BEFORE G12/G13 (lane W2 reads summary.json).
    @{ Id = 'PF1'; Lane = 'W11'; Name = 'Final: C7 with 10 splits, bootstrap 10^4, H2a M = 4 (from checkpoints)'; Kind = 'py'
       Args = @('scripts\confirm\analyze_confirm.py', 'analyze', '--reps', '10', '--out', 'summary_r10.json'); Deps = @('PE1')
       Est = '4-8 h'; Out = 'results\confirm_v1\summary_r10.json' }
    @{ Id = 'PF2'; Lane = 'W11'; Name = 'Final: the 10-split summary becomes summary.json (3-split version kept)'; Kind = 'py'
       Args = @('scripts\confirm\promote_r10.py'); Deps = @('PF1', '~PE2'); Est = '< 1 min'; Out = 'results\confirm_v1\summary.json' }
    @{ Id = 'PF3'; Lane = 'W11'; Name = 'Final: decision rules (SRM) on the new main summary'; Kind = 'py'
       Args = @('scripts\confirm\apply_rules.py', '--no-srnet'); Deps = @('PF2'); Est = '< 1 min'; Out = 'results\confirm_v1\decisions.json' }
    @{ Id = 'PF4'; Lane = 'W11'; Name = 'Final: E5 at the final alpha* (after SRNet; sigma by A5b)'; Kind = 'py'
       Args = @('scripts\experiments\e5_budget.py', '--name', 'e5_summary_confirm.json', '--decisions', 'results\srnet\decisions_srnet.json')
       Deps = @('PF3', 'G13'); Est = '< 1 min'; Out = 'results\e5_budget\e5_summary_confirm.json' }
    @{ Id = 'PF5'; Lane = 'W11'; Name = 'Final: supplementary tables (phases D and E complete)'; Kind = 'py'
       Args = @('scripts\confirm\phase_d.py', 'tables'); Deps = @('PF4', '~D19', '~PE4', '~PE8', '~PE12'); Est = '< 1 min'
       Out = 'results\phase_d\PHASE_D_TABLES_en.md' }
    @{ Id = 'PF6'; Lane = 'W11'; Name = 'Final: Tables 9-12, S1 (RESULTS_TABLES.md)'; Kind = 'py'
       Args = @('scripts\experiments\write_results_confirm.py'); Deps = @('PF5', '~G12'); Est = '< 1 min'; Out = 'results\RESULTS_TABLES.md' }
    @{ Id = 'PF7'; Lane = 'W11'; Name = 'Final: collect results'; Kind = 'py'
       Args = @('scripts\orchestration\collect_results.py'); Deps = @('PF6', '~R1'); Est = '< 1 min'; Out = 'results\manuscript_inputs' }
    @{ Id = 'PF8'; Lane = 'W11'; Name = 'Final: regenerate every figure (results\figures)'; Kind = 'py'
       Args = @('scripts\figures\make_all_figures.py'); Deps = @('PF7'); Est = '2-5 min'; Out = 'results\figures_confirm' }

    # ---------------- W12: post-hoc analyses, manual lane ----------------
    # Not part of the registered protocol; none changes a registered decision (G3 stays failed, alpha* stays at threshold 0.45).
    # PH1 and PH2 read only results\e1_confirm (C3); PH3 reads the final summary and the SRNet decisions (PF3, G13).
    @{ Id = 'PH1'; Lane = 'W12'; Name = 'Post hoc: E1 D4 under the uniform null (R = 1000 per variant)'; Kind = 'py'
       Args = @('scripts\confirm\e1_posthoc_null.py'); Deps = @('C3'); Est = '20-40 min'; Out = 'results\e1_confirm\e1_posthoc_null.json' }
    @{ Id = 'PH2'; Lane = 'W12'; Name = 'Post hoc: E1 D4 interval with model refitting, and its coverage under the null'; Kind = 'py'
       Args = @('scripts\confirm\e1_posthoc_refit.py', '--workers', '7', '--B', '2000', '--null-R', '200', '--null-B', '400'); Deps = @('C3')
       Est = '4-5 h'; Out = 'results\e1_confirm\e1_posthoc_refit.json' }
    @{ Id = 'PH3'; Lane = 'W12'; Name = 'Post hoc: alpha* at decision thresholds 0.35-0.45 (Table S13)'; Kind = 'py'
       Args = @('scripts\analysis\alpha_star_thresholds.py', 'results\confirm_v1\summary.json', '--srnet', 'results\srnet\decisions_srnet.json', '--out', 'results\confirm_v1\alpha_star_thresholds.json')
       Deps = @('PF3', 'G13'); Est = '< 1 min'; Out = 'results\confirm_v1\alpha_star_thresholds.json' }

    # ---------------- W2 ----------------
    @{ Id = 'G1'; Lane = 'W2'; Name = 'Install .venv-gpu (torch CUDA)'; Kind = 'ps'
       Args = @('scripts\gpu\setup_gpu.ps1'); Deps = @(); Est = '10-30 min'; Out = '.venv-gpu'
       Check = { & $PyG -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"; if ($LASTEXITCODE -ne 0) { 'torch finds no CUDA GPU' } } }


    @{ Id = 'G3a'; Lane = 'W2'; Name = 'SRNet G4 HILL 0.4: 200k iterations lr 1e-3 (branch point)'; Kind = 'pyg'
       Args = @('scripts\gpu\srnet_train.py', '--run', 'g4_hill_0.4', '--until', '200000'); Deps = @('C1'); Est = '~11 h'; Out = 'results\srnet\g4_hill_0.4' }
    @{ Id = 'G3b'; Lane = 'W2'; Name = 'SRNet G4 short branch: 50k iterations lr 1e-4'; Kind = 'pyg'
       Args = @('scripts\gpu\srnet_train.py', '--run', 'g4_hill_0.4_short'); Deps = @(); Est = '~3 h'; Out = 'results\srnet\g4_hill_0.4_short' }
    @{ Id = 'G7a'; Lane = 'W2'; Name = 'SRNet G4 verdict (short branch)'; Kind = 'py'
       Args = @('scripts\confirm\srnet_evaluate.py', 'g4'); Deps = @(); Est = '< 5 min'; Out = 'results\srnet\g4_srnet_summary.json'
       Check = { $g = Read-JsonFile 'results\srnet\g4_srnet_summary.json'; if (-not $g) { 'g4_srnet_summary.json missing' } elseif ($g.next -eq 'wait') { 'no result of the short branch yet' } } }
    @{ Id = 'G3c'; Lane = 'W2'; Name = 'Fallback: G4 HILL 0.4 full 500k schedule (only if the short branch fails)'; Kind = 'pyg'
       Args = @('scripts\gpu\srnet_train.py', '--run', 'g4_hill_0.4'); Deps = @(); Est = '~17 h'; Out = 'results\srnet\g4_hill_0.4'
       When = { $g = Read-JsonFile 'results\srnet\g4_srnet_summary.json'; [bool]($g -and $g.next -eq 'fallback_full') }
       WhenText = 'the short branch passed G4: no full schedule needed' }
    @{ Id = 'G7b'; Lane = 'W2'; Name = 'SRNet G4 verdict (full schedule)'; Kind = 'py'
       Args = @('scripts\confirm\srnet_evaluate.py', 'g4'); Deps = @(); Est = '< 5 min'; Out = 'results\srnet\g4_srnet_summary.json'
       When = { (Get-StateOf 'G3c') -eq 'done' }; WhenText = 'full schedule not run'
       Check = { $g = Read-JsonFile 'results\srnet\g4_srnet_summary.json'; if (-not $g -or -not $g.G4_SRNet_pass) { 'SRNet G4 failed with the full schedule as well: confirmatory SRNet not run' } } }
    @{ Id = 'G2'; Lane = 'W2'; Name = 'GPU speed benchmark (idle GPU, before confirmatory SRNet)'; Kind = 'pyg'
       Args = @('scripts\gpu\bench_srnet.py', '--quick'); Deps = @(); Est = '3-5 min'; Out = 'results\gpu_bench' }
    @{ Id = 'G2b'; Lane = 'W2'; Name = 'Decide on 1 or 2 GPU processes'; Kind = 'py'; Soft = $true
       Args = @('scripts\orchestration\gpu_decide.py'); Deps = @(); Est = '< 1 min'; Out = 'results\gpu_bench\decision.json' }
    @{ Id = 'G8'; Lane = 'W2'; Name = 'Confirmatory SRNet REF-U 0.4 (150k iterations)'; Kind = 'pyg'; Soft = $true
       Args = @('scripts\gpu\srnet_train.py', '--run', 'confirm_REF-U_0.4'); Deps = @('G7a', 'G7b', 'C5'); Est = '~8 h (~16 in parallel)'; Out = 'results\srnet\confirm_REF-U_0.4' }
    @{ Id = 'G9'; Lane = 'W2'; Name = 'Confirmatory SRNet REF-U 0.2 (100k iterations)'; Kind = 'pyg'
       Args = @('scripts\gpu\srnet_train.py', '--run', 'confirm_REF-U_0.2'); Deps = @(); Est = '~6 h (~11 in parallel)'; Out = 'results\srnet\confirm_REF-U_0.2' }
    @{ Id = 'G12'; Lane = 'W2'; Name = 'Confirmatory SRNet: P_E, TOST, McNemar'; Kind = 'py'
       Args = @('scripts\confirm\srnet_evaluate.py', 'confirm'); Deps = @('G11'); Est = '< 5 min'; Out = 'results\srnet\confirm_srnet_summary.json' }
    @{ Id = 'G13'; Lane = 'W2'; Name = 'Decision rules with SRNet included'; Kind = 'py'
       Args = @('scripts\confirm\apply_rules.py', '--out', 'results\srnet\decisions_srnet.json'); Deps = @('C8'); Est = '< 1 min'; Out = 'results\srnet\decisions_srnet.json' }

    # ---------------- W3 ----------------
    @{ Id = 'G10'; Lane = 'W3'; Name = 'Confirmatory SRNet PQ-PKS 0.4 (150k iterations)'; Kind = 'pyg'
       Args = @('scripts\gpu\srnet_train.py', '--run', 'confirm_PQ-PKS_0.4'); Deps = @('G7a', 'G7b', 'C5', 'serial:G9'); Est = '~8 h (~16 in parallel)'; Out = 'results\srnet\confirm_PQ-PKS_0.4' }
    @{ Id = 'G11'; Lane = 'W3'; Name = 'Confirmatory SRNet PQ-PKS 0.2 (100k iterations)'; Kind = 'pyg'
       Args = @('scripts\gpu\srnet_train.py', '--run', 'confirm_PQ-PKS_0.2'); Deps = @(); Est = '~6 h (~11 in parallel)'; Out = 'results\srnet\confirm_PQ-PKS_0.2' }

    # ---------------- W4 ----------------
    @{ Id = 'E4'; Lane = 'W4'; Name = 'Full-scale E4 on 512x512 images'; Kind = 'py'
       Args = @('scripts\experiments\e4_functional.py', '--size', '512', '--normal', '100000', '--tamper', '10000', '--cover-images', '900', '--out', 'results\e4_full')
       Deps = @('1', '2'); Est = '25-35 h'; Out = 'results\e4_full\e4_summary.json' }
)
