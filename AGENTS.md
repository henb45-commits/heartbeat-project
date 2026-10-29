# Repository Guide

## Contents

- The dataset is under `ecg-id-database-1.0.0/`.
- It contains 310 ECG recordings from 90 people. Each recording is a 20-second, 500 Hz, two-column signal: raw ECG I and filtered ECG I. Annotation files contain automatically detected R- and T-wave peaks and are unaudited; treat them accordingly.
- Each recording is represented by WFDB `.hea` and `.atr` files. `RECORDS` lists records, and `SHA256SUMS.txt` contains checksums.
- The workspace root also contains a project PDF. Preserve its original filename and contents unless asked to modify it.

## Working Rules

- Treat the recordings and annotations as source data. Do not overwrite, rename, delete, or regenerate them as part of analysis; write derived data to a separate, clearly named location.
- ECG is sensitive biometric and health-related data. Avoid exposing raw signals or participant-level information in logs, reports, screenshots, or committed artifacts unless explicitly needed.
- Check the dataset README and relevant WFDB headers before assuming signal units, record metadata, or annotation semantics.
- Keep analysis reproducible: document inputs, processing choices, and random seeds where applicable.
- No application source code, dependency manifest, or test configuration is currently present. Do not assume build, lint, or test commands; add them only if the project grows code that needs them.