# Week 7 check-in (November 2, 2026)

Team: Hasset Admasu, Aniyah, Adonay, Nathnael

**This week:** the rerun with model selection by validation macro-F1 finished, and we looked at *why* the sequence model stops at about 50% accuracy. We ran two experiments: one on overlapping events, one with a class-weighted loss. The 2022-23 season hold-out is still untouched.

## 1. Final baseline

With models chosen by validation macro-F1, the sequence model is a BiLSTM over frozen DINOv2 frame features. On the 38 test matches it scores 0.498 accuracy, 0.384 macro-F1 and 0.389 balanced accuracy. Always predicting the most common event scores 0.160. This is the baseline for the experiments below. Full numbers are on the [Week 6 page](../2026-10-26_week6/README.md).

## 2. What the model sees

We documented the pipeline from the code and found two problems with the input:
- **The crop.** Each frame is centre-cropped to a square, which keeps only about 43% of a widescreen broadcast frame.
- **Overlapping events.** Each event is a 60-second window (±30 s at 1 fps). 42% of events have another event within 30 seconds, so many windows show more than one event.

## 3. Experiment 1: overlapping events

From the saved test predictions, without retraining:
- Events with another event within 30 seconds are classified 5–24 points worse in 9 of the 14 event types we could compare.
- When a corner happens within 10 seconds, `lead_to_corner` is mistaken for `corner` about 60% of the time. With no corner within 30 seconds, it is about 25%. Most of this confusion comes from the window.
- Events that share an exact timestamp are too few to matter: removing them adds 0.3 points of accuracy.

## 4. Experiment 2: class-weighted loss

We retrained the BiLSTM with rarer classes weighted more heavily in the loss (weights in [results/class_weights.csv](results/class_weights.csv)).

Test results on the 38 held-out matches, mean ± sd over 3 seeds:

| Run | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---|---|
| Baseline | 0.498 ± 0.014 | 0.384 ± 0.018 | 0.389 ± 0.020 |
| Class-weighted | 0.490 ± 0.014 | 0.391 ± 0.006 | 0.413 ± 0.007 |

Macro-F1 here is averaged over the 21 event types that occur in the test matches. The weighted model sometimes predicts `second_yellow_card`, which never occurs in test. The script's report counts that as a 22nd type with F1 = 0 and gives 0.379 ± 0.015 ([results/sequence_classification_report.txt](results/sequence_classification_report.txt)). The baseline never predicts it, so its 0.384 is the same under both conventions.

The weighting shifted errors between classes but did not make the model better overall: the macro-F1 gain is smaller than the variation between seeds. The rarest types, `red_card` and `throw_in`, are still never recognised.

**Conclusion:** the loss is not the main problem; the input is.

## Response to last week's comments

1. **Merge `second_yellow_card` into `red_card`.** Agreed: we will merge them into one "sending off" type, which leaves 21 event types.
2. **Keep `throw_in` separate.** We will keep it and report it as not reliably learnable with the current data. Experiment 2 supports this, since `throw_in` stays at zero even with a class-weighted loss.
3. **Freeze the label set and rerun once.** After the merge, the 21 types are fixed for the rest of the project, and we rerun the baseline once with them.

## Next week

1. Merge `second_yellow_card` into `red_card`, freeze the 21 event types, and rerun the baseline once.
2. Retrain on shorter windows (±5, ±10, ±15 seconds), cut from the existing features.
3. Try encoding the whole frame instead of the centre crop, on a subset of matches.
4. Extend training to 411 matches once the input settings are fixed.

## Questions

1. Should we do the window and crop experiments before extending to 411 matches?

No video, frames or annotations are in this repository. The dataset is under its access agreement.
