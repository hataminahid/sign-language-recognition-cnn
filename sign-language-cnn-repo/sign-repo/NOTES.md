# Notes and known issues

> The notebook keeps the outputs of the original run (Python 3.11.9, Keras 3).
> Only paths/weight loading were edited for portability (hard-coded
> `D:\Arshad\...` paths -> `SIGN_DATA_DIR` / `WEIGHTS_DIR`; offline `.h5` weights ->
> automatic ImageNet download when the files are absent). **Re-run the notebook once
> before publishing** – the edited cells could not be executed where this repository
> was prepared (no dataset, no GPU).

1. **Evaluated weights vs. "best epoch".** `EarlyStopping(restore_best_weights=True)`
   monitors `val_loss`, so the model that is evaluated on the test set carries the
   weights of the *minimum val-loss* epoch (e.g. epoch 44 for EfficientNetB0 stage 2),
   whereas `final_report()` and the report tables quote the epoch of the best
   *val-accuracy* (epoch 35), which is what `ModelCheckpoint` saved. The assignment asks
   for "best model by validation accuracy": to follow it literally, reload
   `efficientnet_stage2_best.keras` / `mobilenet_stage2_best.keras` before evaluating.
2. **Possible near-duplicate leakage.** Train/valid/test were checked for identical
   *file names* only. Roboflow-style names (`K4_jpg.rf.<hash>.jpg`) are augmented
   copies of the same source photo (`K4`) with different hashes, so the same source
   image can appear in several splits without being caught. Compare the part before
   `.rf.` across splits to rule this out.
3. **Tiny evaluation sets.** 72 test images over 26 classes (letters E and L never
   occur in the test split – only 24 classes are scored) and several classes with a
   single test image make precision/recall/F1 very unstable; differences of a few
   points between EfficientNetB0 and MobileNetV2 are within noise.
4. **Section titles.** The notebook headings "Clustering Analysis" follow the
   assignment text, although the task is supervised classification.
5. `docs/technical_report_fa.pdf` shows the student number on its cover page and
   `docs/assignment3_spec.pdf` is the instructor's assignment sheet. Remove or keep
   the repository private if you do not want these public.
6. Training on GPU is not bit-wise deterministic even with fixed seeds
   (`SEED = 42`); expect small run-to-run differences.
