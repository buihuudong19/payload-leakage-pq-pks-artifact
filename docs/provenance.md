# Provenance of the released code

## Frozen profile

Before the confirmatory measurements, the public parameters, the profile identifiers of the systems, the decision rules
and the SHA-256 of every source file were recorded in `configs/profile_frozen_v1.json` (record dated 5 November 2026,
`scripts/freeze_profile.py`). Every confirmatory script calls a check that refuses to run if a file of `src/` or
`configs/` (split files excluded) differs from that record, so all reported measurements were made with the frozen code.
The record lists every source file that existed at freezing time, including files that are not part of this release.

## Changes made for the release

For this release the comments, docstrings, messages and free-text configuration fields were translated into English,
and the code was adapted so that inputs which are not redistributed (the BL1 reference code, the module of
[Zha25], the image datasets) are looked up through documented paths and reported as missing instead of failing. These edits change the
SHA-256 of 33 files of `src/` and `configs/`. The frozen record itself is kept byte for byte, and
`provenance/release_equivalence_v1.json` lists each changed file with its frozen hash, its release hash and the
nature of the change:

| Files | Change | How it was checked |
|---|---|---|
| 25 Python files | comments, docstrings and messages only | identical abstract syntax tree after removing docstrings and replacing every string literal |
| 5 configuration files (`default.yaml`, `g4_reference.json`, `g4_srnet_reference.json`, `srnet_runs_v1.json`, `srnet_runs_v2.json`) | free-text fields and comments only | identical parsed content after replacing every string value |
| `src/pqpks/data.py`, `src/pqpks/crypto/zha25.py`, `src/pqpks/systems/mpvd_systems.py` | lookup of the non-redistributed inputs and the `MissingResource` exception | reviewed by hand; the computations are unchanged |

`scripts/freeze_profile.py --check`, `scripts/confirm/common.check_frozen()` and the check of
`scripts/gpu/srnet_train.py` accept a file whose SHA-256 equals either its frozen hash or the release hash recorded in the
equivalence file for that same frozen hash. Any other change makes the check fail, as before.

Scripts outside `src/` and `configs/` were edited in the same way (translation of comments and messages, removal of
internal notes). Two strings were kept verbatim in Vietnamese because they are written into files whose SHA-256 is
registered: the `note` fields of `configs/splits/e1bis_v1.json` and `configs/splits/e7_v1.json`, produced by
`scripts/confirm/e1bis.py split` and `scripts/confirm/phase_d.py split-e7`. Their English meaning is given in a
comment next to each string.

## Files of the working copy that are not released

Internal planning notes, the design log kept during development, review correspondence, maintenance scripts that only
moved files on the measurement machine, a one-off GPU decision script used once in place of steps G2/G2b, and a script
that reproduced a table of the BL1 reference work are not part of this repository. None of them is called by the
confirmatory pipeline.
