# Public-key Discop [Zha25]

- Source: https://github.com/XinZhang1999/Public-key-Discop, commit `a03633c6fbd343852798a872a9b52e98f8984f8f`,
  file `src/PRNEncryption_P256.py`.
- Article: X. Zhang, K. Chen, N. Zhao, W. Zhang, N. Yu, "Provably Secure Public-Key Steganography Based on Admissible
  Encoding", IEEE Transactions on Information Forensics and Security 20 (2025) 3161-3175.
- The source repository has no license file, so the module is not included here. Run
  `python scripts/setup/fetch_third_party.py` to download it; the expected SHA-256 is in `SHA256SUMS`.
- `src/pqpks/crypto/zha25.py` loads the file unchanged and uses only `encode`, `decode`,
  `PRNEncryption_P256.encode_bytes` and `PRNEncryption_P256.decode_bytes`.
