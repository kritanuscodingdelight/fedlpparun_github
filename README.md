# FedLPPA Prostate MRI Segmentation

## Aim

This project runs FedLPPA, a personalized federated learning framework for weakly supervised medical image segmentation, on prostate MRI data spread across multiple clinical sites. The original paper covers four imaging modalities. This run takes it a step further by adding prostate MRI as a fifth task, working across five federated sites and handling the fact that the I2CVB dataset (Domain 3) was not available.

Paper: FedLPPA: Learning Personalized Prompt and Aggregation for Federated Weakly-supervised Medical Image Segmentation (Lin et al., 2024)
https://arxiv.org/abs/2402.17502

Code Repository: https://github.com/llmir/FedLPPA

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

All volumes were sliced and resized to 384x384. Slices with no prostate visible were dropped. An 80/20 train/test split was applied per site. Sparse annotations across all types including block, keypoint, scribble, scribble noisy, and box were generated from the full masks using morphological operations and skeletonization.

---

## Environment

- Platform: Google Colab with NVIDIA A100 GPU
- Python 3.12
- PyTorch with mixed precision
- Flower 1.4.0 for federated communication

---

## Fixes Applied During Setup

Getting this running on Python 3.12 in a Colab environment required a fair number of patches, both to the dependencies and to the FedLPPA source code itself. Everything below was applied iteratively as errors came up.

### Dependency and Environment Fixes

- Flower 1.4.0 was used instead of 1.0.0 because 1.0.0 caps at Python 3.10 and will not install at all on 3.12. The 1.4.0 wheel works cleanly and keeps the same client and server API that FedLPPA relies on.
- Flower installation kept downgrading numpy to an older version on every run, which broke scipy and scikit-image because those were built against a newer version. The fix was to re-upgrade numpy immediately after Flower finishes installing.
- An older version of grpcio has no pre-built wheel for Python 3.12, so pip tried to compile it from source and failed. Switching to a newer version of grpcio that ships with Python 3.12 wheels resolved this.
- When Flower starts up, it imports tensorflow through one of its internal utility files. This crashes on Colab because the tensorflow version there conflicts with the grpcio version being used. That file was replaced with a stub that does nothing.
- The skeletonize function from scikit-image was swapped out for an equivalent function from OpenCV in the annotation synthesis step because scikit-image pulls in scipy, and importing scipy while numpy is in a broken state causes an unrecoverable crash mid-session. Using OpenCV here sidesteps that entirely.
- Four packages were missing from the repository requirements and needed to be installed separately: efficientnet-pytorch, tensorboardX, info-nce-pytorch, and medpy.
- A CUDA extension included in the repo was compiled for Python 3.9 and simply will not load on Python 3.12. Three files in the repository that import from it were patched with error-catching blocks so the failure is handled gracefully rather than crashing everything. The affected components are set to inactive when the extension is unavailable.
- The error-catching at import time was not enough because one of the affected components actually tries to use the extension at runtime inside the training loop. That component was further patched to skip its operation and return a zero value when the extension is missing.
- Another loss component tries to allocate over 2 GB of memory in a single operation when all five clients are running simultaneously on one GPU. That component was stubbed out to return a zero value, effectively disabling it for this run.

### Training Script Fixes

- The training script has a hard check that rejects any image class value other than three specific options it was originally built for. That check was extended to also accept prostate.
- Data loading workers were set to zero because five clients each spinning up their own worker subprocesses drains the shared memory on Colab, leading to random crashes partway through training.
- The training strategy used here adds three extra loss terms on top of the base cross-entropy loss. Early in training, two of these produce invalid numerical values because the model has not yet learned to segment anything. Each of the three extra loss terms was wrapped in a check so that any invalid value is skipped rather than corrupting the full loss and causing the training round to fail.
- A memory management environment variable was passed to all subprocesses to cut down on GPU memory fragmentation when five clients share a single device.
- All training subprocesses were started in a way that makes them immune to being killed when the notebook cell is stopped. This means the monitoring loop can be interrupted freely without ending the run.
- Server startup detection was broadened to watch for several different log messages rather than just one, since which message appears first depends on how quickly the server comes up.
- Client connection detection was moved to read each client's own log file rather than the server log, because the relevant connection message is only written by the client process itself.

### Dataset and Dataloader Fixes

- A function in the dataset loader always tries to list Domain 3 no matter which client is being set up. Since I2CVB was not available, empty placeholder folders were created for Domain 3 so the listing call does not throw an error.
- Prostate MRI slices come in as single-channel 2D arrays. The evaluation function already handles this correctly through an existing code path that was originally written for a different modality, which adds the necessary dimensions before passing input to the network. No changes were needed here.

---

## Training Configuration

- Strategy: FedUniV2.1
- Model: unet_univ5 with 2.90M parameters
- Max iterations: 3000
- Local iterations per round: 10
- Federated rounds: 300
- Batch size: 6
- Learning rate: 0.003 with polynomial decay
- Mixed precision: enabled
- Image size: 384 by 384
- Number of clients: 5 covering Domains 1, 2, 4, 5, and 6

---

## How to Run

Note: several cells in the notebook were overwritten multiple times as fixes were applied. Only the final version of each cell should be run.

1. The notebook is to be opened in Google Colab with an A100 runtime.
2. The dependency cell is to be run first. The runtime will be automatically restarted once the package versions are pinned to compatible values.
3. After the restart, the sanity check cell is to be run to confirm all imports resolve without errors.
4. Google Drive is to be mounted and the dataset paths are to be checked against the actual folder structure in Drive.
5. The NCI-ISBI preprocessing cell is to be run to produce Domain 1 and Domain 2 data files from the DICOM volumes and mask files.
6. The PROMISE12 preprocessing cell is to be run to produce Domain 4, 5, and 6 data files from the MHD volumes.
7. The annotation synthesis cell is to be run to write all sparse annotation types into each training slice.
8. The verification cell is to be run to check that the data structure looks consistent across all five domains.
9. The symlink cell is to be run to connect the data directory to the path FedLPPA expects.
10. The training cell is to be run. The server is brought up first, then all five clients are launched with a short stagger between each. The display refreshes every 10 seconds with the current round, loss, Dice score, and a rolling history table. Stopping the cell does not end training since all processes run in their own process groups.
11. Checkpoints are written automatically to a folder in Google Drive through a symlink from the FedLPPA model output directory.

---

## Notes

- The paper reports results at 25000 iterations. This run was capped at 3000 due to Colab session time limits. Non-zero Dice scores generally start appearing somewhere around iteration 300 to 500.
- Loss values in the first 50 or so iterations tend to be noisy because the segmentation head starts off predicting all background. This settles down on its own as training continues.
