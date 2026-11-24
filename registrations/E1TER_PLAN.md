# E1-ter: registration (24 November 2026)

The machine-readable registration is `e1ter_plan.json`; `scripts/confirm/e1ter.py` refuses to run when the image list differs
from the SHA-256 recorded there. This note explains the choices.

**Why a third replication.** Gate G3 of E1 failed for the learned distinguisher D4 on BOSSbase/BOWS2, and two post hoc analyses
showed that neither the registered interval (five fitted models kept, too narrow under the null) nor a bootstrap that refits the
models (too wide under the null) calibrates the 0.02 rule. E1-bis passed G3' on fresh ALASKA2 covers, but its bound was one-sided.
No BOSSbase or BOWS2 cover is left unused, so a calibrated test can only be run on the 38 533 ALASKA2 images that no step has used.
E1-ter cannot change the outcome of G3 on BOSSbase/BOWS2.

**Rule.** The null distribution of the advantage is obtained with the five models retrained on every null data set, so the
uncertainty of model fitting is included. The two-sided equivalence rule `|adv| + 1.645 SD_null <= 0.02` bounds both a positive and
a negative advantage (a negative advantage is also a distinguisher once its sign is known). Detection is tested separately against
the same null draws with Holm over the twelve cells.

**Order.** (1) commit this folder and the image list; (2) download and convert the images; (3) split; (4) design step (null
draws, power) - before extraction; (5) extraction; (6) analysis. The script enforces (4) before (5).
