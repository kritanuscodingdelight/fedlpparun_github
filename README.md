# FedLPPA Prostate MRI Segmentation

## Aim

This project runs FedLPPA, a personalized federated learning framework for weakly supervised medical image segmentation, on prostate MRI data spread across multiple clinical sites. The original paper covers four imaging modalities. This run takes it a step further by adding prostate MRI as a fifth task, working across five federated sites and handling the fact that the I2CVB dataset (Domain 3) was not available.

**Paper:** FedLPPA: Learning Personalized Prompt and Aggregation for Federated Weakly-supervised Medical Image Segmentation (Lin et al., 2024)
https://arxiv.org/abs/2402.17502

**Code Repository:** https://github.com/llmir/FedLPPA

---

## Datasets

### NCI-ISBI 2013 Challenge

- Site A (Domain 1): prostate_3t, Siemens 3T scanner, annotation type: block
- Site B (Domain 2): prostate_diagnosis, Philips 1.5T scanner, annotation type: keypoint
- Domain 3 (I2CVB) was not available and was left out. Empty placeholder directories were created so the dataloader does not break when it looks for that folder.

### PROMISE12 Challenge

- Site D (Domain 4): Cases 00-11, Siemens Avanto 1.5T, UCL, annotation type: keypoint
- Site E (Domain 5): Cases 12-22, GE Signa Excite 1.5T, BIDMC, annotation type: scribble noisy
- Site F (Domain 6): Cases 23-49, Siemens Magnetom Trio 3T, Hong Kong, annotation type: block (bounding box converted to block following paper Section III-D)

All volumes were sliced and resized to 384x384. Slices with no prostate visible were dropped. An 80/20 train/test split was applied per site. Sparse annotations across all types (block, keypoint, scribble, scribble noisy, box) were generated from the full masks using OpenCV morphological operations and skeletonization.

---

## Environment

- Platform: Google Colab with NVIDIA A100 GPU
- Python 3.12
- PyTorch with mixed precision (AMP)
- Flower (flwr) 1.4.0 for federated communication

---

## Fixes Applied During Setup

Getting this running on Python 3.12 in a Colab environment required a fair number of patches, both to the dependencies and to the FedLPPA source code itself. Everything below was applied iteratively as errors came up.

### Dependency and Environment Fixes

- flwr 1.4.0 was used instead of 1.0.0 because 1.0.0 caps at Python 3.10 and will not install at all on 3.12. The 1.4.0 wheel works cleanly and keeps the same client/server API that FedLPPA relies on.
- flwr installation kept downgrading numpy to 1.26.4 on every run, which broke scipy and scikit-image because those were built against numpy 2.x. The fix was to re-upgrade numpy to 2.x immediately after flwr finishes installing.
- grpcio 1.46.0 has no pre-built wheel for Python 3.12, so pip tried to compile it from source and failed. Switching to grpcio 1.59.0 resolved this since it ships with Python 3.12 wheels.
- When flwr starts up, it imports tensorflow through `flwr/server/utils/tensorboard.py`. This crashes on Colab because the tensorflow version there conflicts with the grpcio version being used. That file was replaced with a stub that does nothing.
- `skimage.morphology.skeletonize` was swapped out for `cv2.ximgproc.thinning` in the annotation synthesis step because skimage pulls in scipy, and importing scipy while numpy is in a broken state causes an unrecoverable crash mid-session. Using cv2 here sidesteps that entirely.
- Four packages were missing from the repository requirements and needed to be installed separately: `efficientnet-pytorch`, `tensorboardX`, `info-nce-pytorch`, and `medpy`.
- The `tree_filter_cuda` CUDA extension included in the repo was compiled for Python 3.9 and simply will not load on Python 3.12. The three files that import it (`flower_common.py`, `flower_common_v4.py`, `flower_common_v4_addprostate.py`) were patched with try/except blocks so the failure is caught gracefully rather than crashing everything. `TreeEnergyLoss` and `TreeFilter2D` are set to None when the extension is unavailable.
- The try/except at import time was not enough because `TreeEnergyLoss.__init__` actually calls `MinimumSpanningTree` at runtime inside the training loop. The `__init__` was patched to exit early with a disabled flag when the extension is missing, and `forward` was patched to return a zero tensor in that case.
- `ModelLossSemsegGatedCRF.forward` in `gate_crf_loss.py` tries to allocate over 2 GB in a single unfold operation when all five clients are running simultaneously on one GPU. Its forward method was stubbed to return a zero tensor, effectively disabling GatedCRF loss for this run.

### Training Script Fixes

- The training script `flower_pCE_2D_v4_FedLPPA.py` has a hard assert that rejects any `img_class` value other than `odoc`, `faz`, or `polyp`. The assert was extended to accept `prostate` as well.
- DataLoader workers were set to `num_workers=0` because five clients each spinning up their own DataLoader subprocesses drains the shared memory on Colab (`/dev/shm`), leading to random worker crashes partway through training.
- The `FedUniV2.1` strategy adds three losses on top of cross-entropy: GatedCRF, KL divergence on prompts, and pseudo-label Dice loss. Early in training, the pseudo-label Dice produces NaN because the model is predicting all background and Dice on an empty prediction is undefined. The prompt KL also produces NaN on untrained weights. Each of these three loss terms was wrapped in a `torch.isfinite` check so that a NaN result is skipped rather than poisoning the full loss and making the round fail.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was passed to all subprocesses to cut down on GPU memory fragmentation when five clients share a single device.
- All training subprocesses were started with `preexec_fn=os.setsid` so each lives in its own process group. Stopping the notebook cell this way does not send a kill signal to the training processes, which means the monitoring loop can be interrupted freely without ending the run.
- Server startup detection was broadened to watch for several different log strings rather than just one, since which string appears first depends on how quickly the server comes up.
- Client connection detection was moved to read each client's own log file rather than the server log, because the relevant connection string only gets written by the client process itself.

### Dataset and Dataloader Fixes

- The `_get_client_ids_prostate` function in `dataloaders/dataset.py` always tries to list Domain 3 no matter which client is being set up. Since I2CVB was not available, empty `Domain3/train` and `Domain3/test` directories were created as placeholders so the `os.listdir` call does not throw a FileNotFoundError.
- Prostate MRI slices come in as single-channel 2D arrays. The `test_single_volume` function in `val_2D.py` already handles this correctly through its existing faz branch, which adds both the batch and channel dimensions before passing input to the network. No changes were needed here.

---

## Training Configuration

- Strategy: FedUniV2.1
- Model: unet_univ5 (2.90M parameters)
- Max iterations: 3000
- Local iterations per round: 10
- FL rounds: 300
- Batch size: 6
- Learning rate: 0.003 with polynomial decay
- Mixed precision: enabled (AMP)
- Image size: 384x384
- Number of clients: 5 (Domains 1, 2, 4, 5, 6)

---

## How to Run

Note: several cells in the notebook were overwritten multiple times as fixes were applied. Only the final version of each cell should be run.

1. The notebook `FedLPPA_Prostate_Run.ipynb` is to be opened in Google Colab with an A100 runtime.
2. The dependency cell is to be run first. The runtime will be automatically restarted once numpy and scipy are pinned to compatible versions.
3. After the restart, the sanity check cell is to be run to confirm all imports resolve without errors.
4. Google Drive is to be mounted and the dataset paths are to be checked against the actual folder structure in Drive.
5. The NCI-ISBI preprocessing cell is to be run to produce Domain 1 and Domain 2 H5 files from the DICOM volumes and NRRD masks.
6. The PROMISE12 preprocessing cell is to be run to produce Domain 4, 5, and 6 H5 files from the MHD volumes.
7. The annotation synthesis cell is to be run to write all sparse annotation types into each training slice.
8. The verification cell is to be run to check that the H5 structure looks consistent across all five domains.
9. The symlink cell is to be run to connect the data directory to the path FedLPPA expects.
10. The training cell is to be run. The server is brought up first, then all five clients are launched with a short stagger between each. The display refreshes every 10 seconds with the current round, loss, Dice score, and a rolling history table. Stopping the cell does not end training since all processes run in their own process groups.
11. Checkpoints are written automatically to `/content/drive/MyDrive/FedLPPA_prostate_ckpt` through a symlink from the FedLPPA model output directory.

---

## Notes

- The paper reports results at 25000 iterations. This run was capped at 3000 due to Colab session time limits. Non-zero Dice scores generally start appearing somewhere around iteration 300 to 500.
- Loss values in the first 50 or so iterations tend to be noisy because the segmentation head starts off predicting all background. This settles down on its own as training continues.
