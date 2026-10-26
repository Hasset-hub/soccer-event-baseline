# Soccer Event Baseline

Recognizing events in soccer broadcast video with general-purpose visual encoders (CLIP, DINOv2).

**Weekly check-ins:** [checkins/](checkins/README.md) (latest: [Week 6, Oct 26](checkins/2026-10-26_week6/README.md)). The current pipeline is developed in [AcaiLab/Soccer](https://github.com/AcaiLab/Soccer/tree/cleanup/dino-contrastive).

This repository starts from the state described in our last progress report:
- 10 English Premier League 2021-22 matches;
- frames extracted at 1 fps in a ±30 s window around each labeled event (627 events, 18,501 frames);
- per-frame and per-event CLIP / DINOv2 embeddings;
- unsupervised K-means clustering only.

On top of that state, it adds the **supervised mean-pooled baseline** that the feedback asked us to make solid before trying anything more complex:
- a fixed game-level train/val/test split, plus leave-games-out cross-validation;
- one cleaned label list (dropped labels are always reported);
- per-event mean-pooled features;
- linear and MLP classifiers, with and without class weighting;
- macro-F1 as the main metric, plus balanced accuracy, per-class precision/recall/F1 and confusion matrices;
- several seeds, reported as mean ± sd.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt   # baseline + tests only (no torch)
pip install -r requirements.txt       # add this for frame / feature extraction
```

## Try it without any data

```powershell
python scripts/make_synthetic_features.py            # writes features/synthetic/clip_temporal_padded.npz
python run_baseline.py --features features/synthetic
pytest -q
```

The synthetic file has the same layout as the real features: 10 games, padded 60-frame sequences with masks, imbalanced labels including `unknown`, and a per-game visual offset.

## Full pipeline on the real data

1. **Credentials.** Copy `.env.example` to `.env` and fill in the Dropbox access token, the shared-folder link and its password. `.env` is git-ignored, so never commit it.
2. **Videos.** This downloads the `.mkv` halves into `data/videos/<game>/`. Add `--dry-run` to list the files first.
   ```powershell
   python download_data.py
   ```
3. **Annotations.** Put the SoccerReplay-1988 JSON annotations in `data/json/Soccer_Data_Json/train/england_epl_2021-2022/<game>/`. The downloader fetches videos only.
4. **Frames.** This writes `results_by_label_epl_2021_2022/game_NN_half_H/frames/<label>/*.png`.
   ```powershell
   python process_10_games.py
   ```
   By default every event in the JSON is extracted from both halves' videos, as in the original extraction. Add `--filter-half` to keep only each half's own events. Check which behavior matches the annotation format before trusting the labels.
5. **Features and clustering (from the last report).** This produces `features/temporal_dinov2-base/clip_temporal_padded.npz` and the clustering outputs. It runs on CPU and is slow.
   ```powershell
   python run_dino_pipeline.py --model facebook/dinov2-base
   ```
6. **Baseline.**
   ```powershell
   python run_baseline.py --features features/temporal_dinov2-base
   ```

## Baseline outputs (`results/baseline/`)

| File | Contents |
|---|---|
| `labels_report.json` | kept classes; dropped noise labels and rare classes with their event counts |
| `runs.csv` | one row per protocol / fold / model / seed |
| `summary.csv` | mean and sd of macro-F1, balanced accuracy and accuracy per protocol and model |
| `per_class_<model>.csv` | per-class precision, recall, F1 and support on the fixed test games |
| `confusion_<model>.png` | row-normalized confusion matrix on the fixed test games |

Useful options:
- `--pooling mean|max|meanmax`
- `--min-count N`: drop classes with fewer than N events (default 5)
- `--cv-folds K`: 0 runs the fixed split only
- `--seeds 0 1 2`
- `--models ...`

The split lives in `configs/splits.json`: train on games 01–06, validate on 07–08, test on 09–10. Hyperparameters are chosen by validation macro-F1, and the model is then refit on train + val.

## Layout

```
baseline/                 splits, label cleanup, pooling, metrics, synthetic data
configs/splits.json       fixed game-level split
run_baseline.py           supervised baseline
tests/                    pytest suite (runs on synthetic data)
download_data.py          Dropbox video download (credentials from .env)
process_10_games.py       frame extraction for the 10 games
preprocess/               per-video clip / frame extraction
model_encoders.py         CLIP and DINOv2 encoders
temporal_clip_features.py per-event temporal features (E x 60 x D)
extract_frame_features.py flat per-frame features
cluster_analysis.py       K-means + t-SNE across four representations
within_game_clustering.py per-game clustering
run_dino_pipeline.py      extraction + clustering end to end
clustering/*.csv          clustering metrics reported in the last report
```

## Next steps (not in this repo yet)

These follow the feedback, in order:
1. Contrastive projection head trained on training games only, with class-balanced anchor sampling.
2. Embedding-size sweep (64 / 128 / 256 / 512), compared on held-out macro-F1.
3. Random vs semi-hard negatives.
4. Holding out a whole season or competition.
5. Temporal models, compared against this baseline.
