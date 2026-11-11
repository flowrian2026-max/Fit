# Fit By Form — Virtual Try-On Work Log

## 2026-11-11 — FASHN VTON 1.5 weights persisted to Google Drive

### Project links
- GitHub repository (public): https://github.com/flowrian2026-max/Fit
- Colab notebook: https://colab.research.google.com/github/flowrian2026-max/Fit/blob/main/experiments/virtual-tryon/FIT_BY_FORM_FASHN_COLAB.ipynb
- Notebook source on GitHub: https://github.com/flowrian2026-max/Fit/blob/main/experiments/virtual-tryon/FIT_BY_FORM_FASHN_COLAB.ipynb
- Experiment directory: https://github.com/flowrian2026-max/Fit/tree/main/experiments/virtual-tryon
- Private product repository: https://github.com/flowrian2026-max/fit-by-form

### Runtime and model
- Environment: Google Colab
- GPU used in the previously successful smoke test: NVIDIA Tesla T4
- Model: FASHN VTON 1.5
- Upstream source revision pinned by the runner: `7c0f10af3f91ad4048fe9729c470a13ef905d25a`
- Previous real GPU smoke test completed successfully with 30 steps and produced `tryon-output.png` and `tryon-metrics.json`. This confirms a technical smoke test only; production quality/readiness has not been established.

### Confirmed weight files in the current Colab runtime
- `weights/` contained 13 files; total reported size: 2.14 GB.
- `weights/model.safetensors`: 1.9G.
- `weights/dwpose/`: 335M.
- These checks confirmed that the model weights were already present in the active runtime; no re-download was needed for the copy.

### Google Drive persistence
- Drive mount point: `/content/drive`
- Destination folder: `/content/drive/MyDrive/Fit-Model-Weights/`
- Copied model directory: `/content/drive/MyDrive/Fit-Model-Weights/weights/`
- Copy command used:
  ```bash
  !mkdir -p /content/drive/MyDrive/Fit-Model-Weights
  !cp -a weights /content/drive/MyDrive/Fit-Model-Weights/
  ```
- Verification output after copy: `2.2G /content/drive/MyDrive/Fit-Model-Weights/weights`.
- Status: copy command completed and destination directory size was reported. Individual destination files have not yet been independently checked.

### Next actions
1. Verify `model.safetensors` and the `dwpose` directory at the Google Drive destination.
2. Update the Colab notebook so subsequent runs use `/content/drive/MyDrive/Fit-Model-Weights/weights` as the weights directory and do not download the model again when the files are present.
3. Run a lightweight path/weight validation before considering another inference run.
4. Do not upload the 2.2 GB model weights to the public GitHub repository. Keep model weights in Google Drive or another suitable persistent model-storage location.

### Important status distinction
- Notebook and helper scripts are stored in the public `Fit` repository on `main`.
- Model weights are **not confirmed to be stored in GitHub**.
- A copy of the weights has been made to the Google Drive folder above; post-copy file-level verification remains pending.


## 2026-11-11 — Colab notebook updated for Drive-first model loading

- Updated notebook: https://github.com/flowrian2026-max/Fit/blob/main/experiments/virtual-tryon/FIT_BY_FORM_FASHN_COLAB.ipynb
- The notebook now mounts Google Drive before setup.
- Persistent weights path: `/content/drive/MyDrive/Fit-Model-Weights/weights`.
- If both `model.safetensors` and `dwpose/` are present in Drive, the notebook links the local `weights/` path to Drive and skips the ~2 GB download.
- If Drive weights are absent, the notebook downloads weights once, copies them to Drive, and uses the local copy for that run.
- Notebook update committed to `main`: `208f485e4f539cd97fcfd1acb0727f350282df7d`.
- Remaining verification: run the updated notebook from a fresh Colab runtime and confirm it reports that it is using saved Google Drive weights. This verification has not yet been performed.
