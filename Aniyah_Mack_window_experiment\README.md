# Week 8 -- Aniyah Mack: window experiment

Tested shorter windows around each event (cut from the saved DINOv2 features) and how much
look-ahead after the event the BiLSTM needs. Validation only, 3 seeds; the test set was not used.

## Your parts of the code
- `soccer/features.py`: `TemporalFeatures.window()` and `frame_offsets()` (the rest of the file is shared).
- `scripts/sequence_classifier.py`: the `--before-sec`, `--after-sec`, `--frames-root`, `--models`, `--val-only` options.
- `tests/test_features_metrics.py`: `test_window_keeps_frames_in_range_packed_to_the_front`, `test_frame_offsets_from_frame_filenames`.
- `tests/test_scripts_smoke.py`: `test_sequence_classifier_val_only_writes_no_test_results`.
- `changes.patch`: the full diff of these files (shared files include other members' changes too).

## Results (`results/exp_window/`, one folder per window; `run.sh` is how it was run)
| Window (before / after, seconds) | Val macro-F1 (mean +/- sd) | Val accuracy |
|---|---|---|
| 30 / 30 (current) | 0.387 +/- 0.010 | 0.479 |
| 30 / 10 | 0.351 +/- 0.010 | 0.450 |
| 30 / 5 | 0.333 +/- 0.011 | 0.443 |
| 30 / 0 | 0.286 +/- 0.006 | 0.384 |
| 15 / 15 | 0.336 +/- 0.009 | 0.447 |
| 10 / 10 | 0.319 +/- 0.008 | 0.417 |

Shorter windows are worse, so the full window stays. About 10 s of look-ahead keeps most of the accuracy.
