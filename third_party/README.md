# third_party

Third-party code used by a baseline. It is not redistributed; `scripts/setup/fetch_third_party.py` downloads it from
its public repository at a fixed commit and checks the SHA-256 recorded here.

| Folder | Source | Used for |
|---|---|---|
| `zha25_public_key_discop/` | https://github.com/XinZhang1999/Public-key-Discop, commit `a03633c6fbd343852798a872a9b52e98f8984f8f`, file `src/PRNEncryption_P256.py` | BL4: admissible point encoding of [Zha25] (see `docs/baselines.md`) |
