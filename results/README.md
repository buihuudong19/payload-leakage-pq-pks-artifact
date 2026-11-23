# results

Every script writes its output below this folder. Only `e5_budget/e5_ladder.json` is tracked: it is an input of the
decision rules (`scripts/confirm/apply_rules.py`) and of the tests, and can be regenerated with
`python scripts/experiments/e5_ladder.py`.

The measurement data of the article (result summaries, per-cell warden scores, extracted strings, E1 and E1-bis
syndromes, SRNet scores) are in the Zenodo dataset record `pq-pks-raw-data-v1`. Copying the `results/` folder of that
archive (`pq-pks-raw-data-v1/results/`) into the repository root restores the corresponding subfolders.
