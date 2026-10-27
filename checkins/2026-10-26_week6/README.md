# Week 6 check-in (October 26, 2026)

Team: Hasset Admasu, Aniyah, Adonay, Nathnael

Written report: [week6_report.tex](week6_report.tex) (LaTeX).

**This week:** baseline results on real data, scored with macro-F1 and balanced accuracy, on 255 Premier League matches (up from 65). The 2022-23 season is reserved, untouched, as the final hold-out. Model selection now uses macro-F1. A rerun with that selection is in progress, and this page will be updated when it finishes.

## 1. Baseline results on real data

The data:
- **Matches:** 255 Premier League matches from 2017-18 to 2019-20, split as in SoccerReplay-1988: 182 train / 35 val / 38 test. No match appears in two splits.
- **Events:** 22 event types kept, 15,860 events (train 11,243 / val 2,307 / test 2,310). Four types are excluded (718 events): two broadcast graphics and two annotation artefacts.
- **Halves left out:** six halves failed the frame-timing check, so their 203 events are left out.

Test results on the 38 held-out matches, mean ± sd over 3 seeds. The interval is the 95% range from resampling whole test matches.

| Encoder | Model | Accuracy | Macro-F1 | Balanced accuracy | 95% interval (accuracy) |
|---|---|---|---|---|---|
| — | Majority class | 0.160 | — | — | — |
| DINOv2 | Pooled features + best classifier | 0.410 ± 0.008 | 0.278 ± 0.002 | 0.273 ± 0.007 | 0.389–0.432 |
| DINOv2 | Sequence model | 0.509 ± 0.005 | 0.390 ± 0.009 | 0.393 ± 0.013 | 0.486–0.533 |
| MatchVision | Pooled features + best classifier | 0.510 ± 0.012 | 0.385 ± 0.019 | 0.378 ± 0.016 | 0.491–0.531 |
| MatchVision | Sequence model | 0.523 ± 0.007 | 0.405 ± 0.001 | 0.406 ± 0.008 | 0.502–0.546 |

- **DINOv2** is a general-purpose image encoder, frozen.
- **MatchVision** (UniSoccer) is a video encoder pretrained on soccer broadcasts, also frozen.
- **Pooled features:** we average the 60 frame embeddings around each event, then try 15 candidates (5 feature sets × 3 classifiers).
- **Sequence model:** reads the 60 frames in order. The candidates are a BiLSTM, a temporal CNN, a Transformer and a mean-pool MLP.

Every model is chosen on validation matches and scored once on test. **The models in this table were chosen by validation accuracy.** The rerun that chooses by validation macro-F1 is in progress (section 4).

Full reports: [results/report_dinov2.md](results/report_dinov2.md), [results/report_matchvision.md](results/report_matchvision.md).

## 2. Rare classes: kept separate, with counts and per-class F1

Following the feedback, no classes were merged. The table shows the rarest types first. F1 is on test, as the mean over seeds.

| Event type | Train | Val | Test | DINOv2 sequence F1 | MatchVision sequence F1 |
|---|---|---|---|---|---|
| second_yellow_card | 8 | 3 | 0 | no test events | no test events |
| red_card | 20 | 5 | 3 | 0.000 | 0.000 |
| throw_in | 28 | 14 | 7 | 0.000 | 0.000 |
| penalty | 39 | 8 | 9 | 0.063 | 0.257 |
| foul_lead_to_penalty | 68 | 14 | 15 | 0.292 | 0.271 |
| var | 71 | 12 | 25 | 0.445 | 0.415 |
| ball_out_of_play | 168 | 37 | 30 | 0.097 | 0.016 |
| ball_possession | 365 | 63 | 92 | 0.191 | 0.236 |
| goal | 624 | 125 | 126 | 0.673 | 0.668 |
| substitution | 997 | 192 | 211 | 0.741 | 0.767 |

The other types are in [results/per_class_dinov2.csv](results/per_class_dinov2.csv) and [results/per_class_matchvision.csv](results/per_class_matchvision.csv), with class counts per split in [results/label_counts.csv](results/label_counts.csv).

What we see:
- **The rarest types are not learned.** The three types with fewer than 30 training events are `second_yellow_card` (8), `red_card` (20) and `throw_in` (28). `red_card` and `throw_in` get F1 = 0 with both encoders and both models. `second_yellow_card` has no test events at all, so it cannot be scored.
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
- **MatchVision caveat:** MatchVision was pretrained on SoccerReplay-1988's official training matches, so 89 of the 115 hold-out matches may have been in its pretraining data. We will report MatchVision on all 115 matches and also on the 26 official val/test matches only. DINOv2 was not trained on this dataset.

Details: [docs/season_holdout.md](https://github.com/AcaiLab/Soccer/blob/cleanup/dino-contrastive/docs/season_holdout.md).

## 4. Model selection now uses macro-F1

Until this week, the classifiers were chosen by validation accuracy even though macro-F1 is our main metric. The two can disagree. On the MatchVision pooled features, for example:
- **Best on accuracy:** max-pool + MLP (validation macro-F1 0.347).
- **Best on macro-F1:** mean-pool + linear SVM (validation macro-F1 0.363).

Every classifier now ranks candidates by validation macro-F1, with accuracy breaking ties. The sequence models also stop training early based on it. The test matches are still used only once. The rerun on all 255 matches is in progress, and the table in section 1 will be updated with its results.

## Code this week

All in [AcaiLab/Soccer, branch `cleanup/dino-contrastive`](https://github.com/AcaiLab/Soccer/tree/cleanup/dino-contrastive):

| Commit | Change |
|---|---|
| [50f01cd](https://github.com/AcaiLab/Soccer/commit/50f01cd) | Manifest builder for whole league seasons; manifest for 2017-20 (255 matches) |
| [6b08426](https://github.com/AcaiLab/Soccer/commit/6b08426) | MatchVision per-frame features in the same format as DINOv2's |
| [092ada6](https://github.com/AcaiLab/Soccer/commit/092ada6) | MatchVision config; leave out halves that fail the frame-timing check |
| [766c1de](https://github.com/AcaiLab/Soccer/commit/766c1de) | Choose models by validation macro-F1 |
| [ae68b9c](https://github.com/AcaiLab/Soccer/commit/ae68b9c) | Per-class F1 next to class counts, in every report |
| [6eceac8](https://github.com/AcaiLab/Soccer/commit/6eceac8) | Reserve 2022-23 (115 matches) as the season hold-out |

No video, frames or annotations are in either repository. The dataset is under its access agreement.

## Next week

1. Finish the macro-F1 rerun and update section 1.
2. Decide on the rare classes (question 1 below), then rerun once.
3. Extract 2020-21 and 2021-22 together with the 255 matches, so training uses 411 matches.

## Questions

1. Is merging `second_yellow_card` into `red_card` ("sending off") acceptable, with `throw_in` kept separate and reported as not learnable?
2. For MatchVision on the hold-out season, is reporting both all 115 matches and the 26 matches outside its pretraining data the right way to handle the overlap?
