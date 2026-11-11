# Week 6 check-in (October 26, 2026)

Team: Hasset Admasu, Aniyah, Adonay, Nathnael

**This week:** baseline results on real data, scored with macro-F1 and balanced accuracy, on 255 Premier League matches (up from 65). The 2022-23 season is reserved, untouched, as the final hold-out. Model selection now uses macro-F1, and every result on this page comes from that selection.

## 1. Baseline results on real data

The data:
- **Matches:** 255 Premier League matches from 2017-18 to 2019-20, split as in SoccerReplay-1988: 182 train / 35 val / 38 test. No match appears in two splits.
- **Events:** 22 event types kept, 15,860 events (train 11,243 / val 2,307 / test 2,310). Four types are excluded (718 events): two broadcast graphics and two annotation artefacts.
- **Halves left out:** six halves failed the frame-timing check, so their 203 events are left out.

Test results on the 38 held-out matches, mean ± sd over 3 seeds. The interval is the 95% range from resampling whole test matches.

| Model | Accuracy | Macro-F1 | Balanced accuracy | 95% interval (accuracy) |
|---|---|---|---|---|
| Majority class | 0.160 | — | — | — |
| Pooled features + best classifier | 0.380 ± 0.033 | 0.281 ± 0.003 | 0.278 ± 0.010 | 0.356–0.403 |
| Sequence model (BiLSTM) | 0.498 ± 0.014 | 0.384 ± 0.018 | 0.389 ± 0.020 | 0.477–0.522 |

- **Features:** every frame is encoded with DINOv2, a general-purpose image encoder, kept frozen.
- **Pooled features:** we average the 60 frame embeddings around each event, then try 15 candidates (5 feature sets × 3 classifiers).
- **Sequence model:** reads the 60 frames in order. The candidates are a BiLSTM, a temporal CNN, a Transformer and a mean-pool MLP; the BiLSTM is chosen. It leads the pooled features by +0.118 accuracy, in all 3 seeds.

Every model is chosen on validation matches by macro-F1 and scored once on test.

Full report: [results/report_dinov2.md](results/report_dinov2.md).

## 2. Rare classes: kept separate, with counts and per-class F1

Following the feedback, no classes were merged. The table shows the rarest types first. F1 is on test, as the mean over seeds.

| Event type | Train | Val | Test | Sequence model F1 |
|---|---|---|---|---|
| second_yellow_card | 8 | 3 | 0 | no test events |
| red_card | 20 | 5 | 3 | 0.000 |
| throw_in | 28 | 14 | 7 | 0.000 |
| penalty | 39 | 8 | 9 | 0.063 |
| foul_lead_to_penalty | 68 | 14 | 15 | 0.210 |
| var | 71 | 12 | 25 | 0.445 |
| ball_out_of_play | 168 | 37 | 30 | 0.096 |
| ball_possession | 365 | 63 | 92 | 0.184 |
| goal | 624 | 125 | 126 | 0.671 |
| substitution | 997 | 192 | 211 | 0.743 |

The other types are in [results/per_class_dinov2.csv](results/per_class_dinov2.csv), with class counts per split in [results/label_counts.csv](results/label_counts.csv).

What we see:
- **The rarest types are not learned.** The three types with fewer than 30 training events are `second_yellow_card` (8), `red_card` (20) and `throw_in` (28). `red_card` and `throw_in` get F1 = 0 with both models. `second_yellow_card` has no test events at all, so it cannot be scored.
- **Rarity is not the whole story.** `ball_out_of_play` has 168 training events and `ball_possession` has 365, yet both stay below 0.25. These types look alike on screen, so more data alone may not fix them.
- **Proposal, not applied:** merge `second_yellow_card` into `red_card` as one "sending off" type. A second yellow card is a sending off, and the two together still have only 28 training events. We see no meaningful broader category for `throw_in`, so we would keep it separate and report it as not yet learnable.

## 3. Season hold-out: 2022-23 reserved

Premier League matches in the dataset, by season:

| Season | Matches | Official train / val / test | Used in development |
|---|---|---|---|
| 2017-18 to 2019-20 | 255 | 182 / 35 / 38 | yes (this week) |
| 2020-21 | 91 | 68 / 13 / 10 | no |
| 2021-22 | 65 | 50 / 7 / 8 | yes (weeks 2–5) |
| **2022-23** | **115** | **89 / 13 / 13** | **no: reserved** |
| 2023-24 | 26 | 21 / 3 / 2 | no |

- **Why 2022-23:** it is the most recent season with enough matches for a test set, and it has never been used. 2023-24 is newer but has only 26 matches.
- **How it is reserved:** the 115 matches are listed in a manifest with every match marked `test`. Nothing reads their frames, features or labels until the final models are frozen.
- **Training data:** we will train and validate on 2017-18 to 2021-22 (411 matches).

Details: [docs/season_holdout.md](https://github.com/AcaiLab/Soccer/blob/cleanup/dino-contrastive/docs/season_holdout.md).

## 4. Model selection now uses macro-F1

Until this week, the classifiers were chosen by validation accuracy even though macro-F1 is our main metric, and the two can disagree. Every classifier now ranks candidates by validation macro-F1, with accuracy breaking ties. The sequence models also stop training early based on it. The test matches are still used only once.

The rerun on all 255 matches is finished. Compared with selection by accuracy:
- **Sequence model:** the BiLSTM is chosen under both rules. Only the stopping epoch changes, and test macro-F1 moves from 0.390 ± 0.009 to 0.384 ± 0.018, within the spread between seeds.
- **Pooled features:** seed 42 still chooses mean pooling + MLP; seeds 43 and 44 now choose mean pooling + logistic regression. Test macro-F1 moves from 0.278 to 0.281, while accuracy falls from 0.410 to 0.380.

Choosing by macro-F1 did not raise test macro-F1 measurably, but the models are now chosen on the metric we report.

## Code this week

All in [AcaiLab/Soccer, branch `cleanup/dino-contrastive`](https://github.com/AcaiLab/Soccer/tree/cleanup/dino-contrastive):

| Commit | Change |
|---|---|
| [50f01cd](https://github.com/AcaiLab/Soccer/commit/50f01cd) | Manifest builder for whole league seasons; manifest for 2017-20 (255 matches) |
| [092ada6](https://github.com/AcaiLab/Soccer/commit/092ada6) | Leave out halves that fail the frame-timing check |
| [766c1de](https://github.com/AcaiLab/Soccer/commit/766c1de) | Choose models by validation macro-F1 |
| [ae68b9c](https://github.com/AcaiLab/Soccer/commit/ae68b9c) | Per-class F1 next to class counts, in every report |
| [6eceac8](https://github.com/AcaiLab/Soccer/commit/6eceac8) | Reserve 2022-23 (115 matches) as the season hold-out |

No video, frames or annotations are in either repository. The dataset is under its access agreement.

## Next week

1. Decide on the rare classes (question below), then rerun once.
2. Extract 2020-21 and 2021-22 together with the 255 matches, so training uses 411 matches.

## Questions

1. Is merging `second_yellow_card` into `red_card` ("sending off") acceptable, with `throw_in` kept separate and reported as not learnable?
