# Recorded experiment: 20261011T060628Z_5a34e14b

Curated from the completed Kaggle run. All six trials finished 900 updates.

- `*_final.pt`: complete model/optimizer checkpoints, including scales, history and RNG states.
- `checkpoint_inventory.csv`: file sizes and original SHA-256 checksums for the six retained checkpoints.
- `run_metadata.json`: exact trial definitions, preprocessing and recorded environment.
- `comparison.csv` and `paired_comparison.csv`: full metrics and matched MLP/PI-MLP comparisons.
- `all_models_residuals.csv`: all 4096 timestamps for each of the six trials (24,576 rows).
- `*_history.csv`: optimization history for each trial.
- `*_prediction_and_residual.png`: full/test predictions and signed residuals, one figure per trial.
- Other PNGs: residual maps, local RMSE, learning curves, and architecture/family score comparison.
- `best_trial.txt`: trial with the lowest test RMSE under the coursework selection rule.

The combined residual CSV already contains the per-model prediction tables, so their
duplicate individual CSVs are omitted. The original ZIP's latest checkpoints duplicate
the final files; latest and previous copies are also omitted. The six final checkpoints
and unchanged metadata are sufficient for the notebook's completed-run restore path.

Each final checkpoint's checksum was checked against the original archive inventory.
No model was retrained while preparing this repository. See the root README for data
setup, checkpoint loading, experiment limits, and data attribution.
