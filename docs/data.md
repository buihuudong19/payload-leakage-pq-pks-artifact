# Data

The experiments use three external image sets. None of their images is included in this repository or in the
dataset record of the article, because their terms of use do not allow redistribution (or do not grant it explicitly).
What is included is everything needed to rebuild exactly the same inputs: the image lists, the frozen splits and the
SHA-256 of every image used.

## External datasets

| Dataset | Used in | Where to obtain it | Terms |
|---|---|---|---|
| BOSSbase 1.01, 10 000 grayscale 512x512 PGM images | E1, E2/E3, E4, E6, SRNet, gate G4, pilot and screening rounds | BOSS website (http://agents.fel.cvut.cz/boss/) or the Binghamton image database (https://dde.binghamton.edu/download/ImageDB/) | terms of the providers; cite [BFP11] |
| BOWS2-Original, 10 000 grayscale 512x512 PGM images (numbered 0-10000, 3661 missing) | E1, E2/E3 training part, SRNet, screening round | Binghamton image database (https://dde.binghamton.edu/download/ImageDB/), file `BOWS2OrigEp3.tgz` | scientific use only, no commercial use, cite as "BOWS2-Original" (licence file distributed with the archive) |
| ALASKA2, uncompressed grayscale 512x512 base images | E1-bis (40 000 images), E7 (10 000 images) | https://alaska.utt.fr (download script `ALASKA_v2_TIFF_512_GrayScale.sh`; camera list `ALASKAv2Base_by_DSC.txt`) | Creative Commons BY-NC-ND 4.0; cite [CGB20] |

The copies used for the article were obtained as `BOSSbase_1.01.zip` and `bows2ep3.zip` (an archive that contains the
original `BOWS2OrigEp3.tgz`); their SHA-256 are in `data/external/raw/SHA256SUMS`. Other copies of the same releases
should give the same per-image hashes; this can be checked with the manifests below.

## What the repository contains

```
data/external/
  _pilot/pilot_bossbase_ids.txt     1000 BOSSbase pilot images (seed 20261029)
  _pilot/screen_bows2_ids.txt       1000 BOWS2 screening images (seed 20261105)
  bossbase/MANIFEST.sha256          SHA-256 of the 10 000 BOSSbase PGM files
  bows2/MANIFEST.sha256             SHA-256 of the 10 000 BOWS2 PGM files
  alaska2/subset_ids.txt            E7: 10 000 ALASKA2 images (index, camera model), stratified by camera model
  alaska2/e1bis_ids.txt             E1-bis: 40 000 ALASKA2 images (the E7 subset + 30 000 additional images)
  alaska2/MANIFEST_tiff.sha256      SHA-256 of the 40 000 downloaded ALASKA2 TIFF files
  raw/SHA256SUMS                    SHA-256 of the downloaded BOSSbase and BOWS2 archives
configs/splits/
  split_v1.json (+ split_v1.sha256) frozen split of the 20 000 BOSSbase + BOWS2 covers (E1 folds, SRM test sets)
  e1bis_v1.json                     5 folds of E1-bis
  e7_v1.json                        3 splits of E7, stratified by camera model
```

The SHA-256 of `split_v1.json`, `e1bis_ids.txt`, `e1bis_v1.json` and `e7_v1.json` are registered in
`scripts/orchestration/pipeline.ps1`; the corresponding steps fail if a regenerated file differs. These files are stored
byte for byte (`.gitattributes` disables line-ending conversion for them).

## Preparing the images

1. Download the archives and place them in `data/external/raw/`.
2. Extract BOSSbase into `data/external/bossbase/` (files `1.pgm` ... `10000.pgm`) and BOWS2 into
   `data/external/bows2/` (files `0.pgm` ... `10000.pgm`, without `3661.pgm`).
3. Check the images:
   ```bash
   python scripts/setup/verify_manifest.py data/external/bossbase
   python scripts/setup/verify_manifest.py data/external/bows2
   ```
4. ALASKA2 (Windows): `powershell -ExecutionPolicy Bypass -File scripts\data\download_alaska2.ps1 -List data\external\alaska2\e1bis_ids.txt`,
   then `python scripts/data/verify_alaska2.py --list data/external/alaska2/e1bis_ids.txt`, which checks every TIFF,
   records its SHA-256 and writes the PGM copies into `data/external/alaska2/pgm/`.

Other locations can be given with `PQPKS_BOSSBASE_DIR`, `PQPKS_BOWS2_DIR` and `PQPKS_ALASKA2_DIR`.

Notes from the preparation of the article:

- BOSSbase contains four pairs of byte-identical images (5341/5536, 5342/5537, 5343/5538, 5344/5539) and BOWS2 five
  (6532/6533, 7175/7263, 7177/7265, 7577/7602, 7808/7856). `scripts/confirm/make_split.py` groups byte-identical and
  near-duplicate images so that each group falls into one part of the split.
- About 7 % of the BOWS2 files have a comment line in the PGM header; the loaders read them with PIL, not at a fixed offset.
- 21 of the ALASKA2 TIFF files are stored as a single strip (262 314 bytes instead of 262 570); `verify_alaska2.py`
  decodes every file, so both layouts are accepted.

## Test covers

The unit tests and `scripts/demo_pqpks_flow.py` use eight synthetic 512x512 grey-scale covers and two binary test
messages (919 and 2420 bytes) generated deterministically by `pqpks.data.cover_pool()` and `pqpks.data.test_messages()`.
No image file is needed. E4 (`scripts/experiments/e4_functional.py`) uses the same two binary messages for one message
in ten; all its covers come from BOSSbase and BOWS2. No other experiment of the article uses the synthetic data.

## References

- [BFP11] P. Bas, T. Filler, T. Pevný, "Break our steganographic system": the ins and outs of organizing BOSS,
  Information Hiding (IH 2011), LNCS 6958, pp. 59-70, 2011. doi:10.1007/978-3-642-24178-9_5
- [CGB20] R. Cogranne, Q. Giboulot, P. Bas, ALASKA-2: challenging academic research on steganalysis with realistic
  images, IEEE International Workshop on Information Forensics and Security (WIFS), 2020. doi:10.1109/WIFS49906.2020.9360896
