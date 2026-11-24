# registrations

Plans of the analyses added after the review of 24 November 2026, fixed before the analyses were run. The commit that adds
a plan to the repository is its registration date. Each script refuses to run when its input differs from the SHA-256
recorded in the plan.

| File | Analysis | Script |
|---|---|---|
| `g4_new_seeds_plan.json` | gate G4 for SRM on ten fresh seeds | `scripts/confirm/g4_new_seeds.py` |
| `e1ter_plan.json`, `E1TER_PLAN.md` | E1-ter: calibrated two-sided test of D4 on 38 533 unused ALASKA2 images | `scripts/confirm/e1ter.py` |
