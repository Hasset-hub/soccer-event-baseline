"""
Sequence models over per-frame frozen features (DINOv2 or MatchVision) for event classification.

Instead of pooling the 60-frame window into a single vector, these models
consume the full (60, 768) sequence, using temporal ordering:

    1. MeanPool-MLP  : masked mean over frames -> MLP        (baseline)
    2. BiLSTM        : bidirectional LSTM -> masked mean of hidden states
    3. TemporalCNN   : stacked 1D convolutions over time -> masked mean
    4. Transformer   : self-attention over time with a padding mask

Protocol (leak-free, by game, from the manifest):
    train games fit the weights, val games pick the stopping epoch and the
    model (both by macro-F1, accuracy breaking ties), and only the chosen
    model is scored on the test games. With --seeds, every model is trained
    once per seed, the model with the best mean validation macro-F1 is
    chosen, and its test scores are reported as mean and standard deviation
    over seeds plus a 95% interval from resampling test games.
Per-dimension standardisation uses training frames only.

Usage:
    python scripts/sequence_classifier.py
    python scripts/sequence_classifier.py --before-sec 30 --after-sec 5   # less look-ahead
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score

from soccer.config import data_path, dropped_labels, label_merges, load_config, repo_path
from soccer.encoding import pick_device
from soccer.evaluation import prediction_rows, split_by_game
from soccer.features import frame_offsets, load_temporal
from soccer.manifest import load_manifest, manifest_games
from soccer.metrics import bootstrap_ci, classification_scores


def masked_mean(h, m):
    """h: (B, T, D), m: (B, T) -> (B, D)"""
    return (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)


class MeanPoolMLP(nn.Module):
    def __init__(self, d_in, d_h, n_cls, max_len):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d_h, n_cls))

    def forward(self, x, m):
        return self.net(masked_mean(x, m))


class BiLSTM(nn.Module):
    def __init__(self, d_in, d_h, n_cls, max_len):
        super().__init__()
        self.proj = nn.Linear(d_in, d_h)
        self.lstm = nn.LSTM(d_h, d_h, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d_h, n_cls))

    def forward(self, x, m):
        # Pack so the backward direction starts at each event's last real frame, not in its
        # padding (real frames come first; events at the start of a half have fewer).
        lengths = m.sum(1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(torch.relu(self.proj(x)), lengths, batch_first=True,
                                                   enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        return self.head(masked_mean(h, m))


class TemporalCNN(nn.Module):
    def __init__(self, d_in, d_h, n_cls, max_len):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_in, d_h, 5, padding=2), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(d_h, d_h, 5, padding=2), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(d_h, n_cls)

    def forward(self, x, m):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, d_h)
        return self.head(masked_mean(h, m))


class TransformerClassifier(nn.Module):
    def __init__(self, d_in, d_h, n_cls, max_len, n_heads=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_h)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_h) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_h, nhead=n_heads, dim_feedforward=d_h * 4,
                                           dropout=0.3, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_h, n_cls))

    def forward(self, x, m):
        h = self.proj(x) + self.pos[:, :x.size(1)]
        h = self.encoder(h, src_key_padding_mask=~m.bool())
        return self.head(masked_mean(h, m))


MODELS = {"MeanPool-MLP": MeanPoolMLP, "BiLSTM": BiLSTM, "TemporalCNN": TemporalCNN,
          "Transformer": TransformerClassifier}


def main(argv=None):
    cfg = load_config()
    prefix = cfg["features"]["model"].split("/")[-1]
    parser = argparse.ArgumentParser(description="Sequence classifiers over per-frame features.")
    parser.add_argument("--features", default=str(data_path(cfg, "features") / f"temporal_{prefix}"))
    parser.add_argument("--out-dir", default=str(data_path(cfg, "results") / "classification"))
    parser.add_argument("--val-games", nargs="+", default=None, help="default: manifest val split")
    parser.add_argument("--test-games", nargs="+", default=None, help="default: manifest test split")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[cfg["seed"]])
    parser.add_argument("--keep-all-labels", action="store_true")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS),
                        help="which sequence models to train (default: all)")
    parser.add_argument("--val-only", action="store_true",
                        help="stop after validation: no test scores, e.g. while choosing a setting")
    parser.add_argument("--before-sec", type=float, default=None,
                        help="keep only frames this many seconds before the event (default: all)")
    parser.add_argument("--after-sec", type=float, default=None,
                        help="keep only frames this many seconds after the event (default: all)")
    parser.add_argument("--frames-root", default=str(data_path(cfg, "frames")),
                        help="frame folders, to rebuild frame times for --before-sec/--after-sec")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto (GPU only if it has free memory)")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf = load_temporal(args.features, games=manifest_games(cfg))
    if args.before_sec is not None or args.after_sec is not None:
        offsets, mismatched = frame_offsets(tf, args.frames_root)
        if mismatched:
            raise SystemExit(f"{mismatched} events have a different number of frames on disk than "
                             f"encoded; their frame times are unknown, so a window cannot be applied")
        before = np.inf if args.before_sec is None else args.before_sec
        after = np.inf if args.after_sec is None else args.after_sec
        n_full = tf.masks.sum(1).mean()
        tf = tf.window(offsets, before, after)
        print(f"Window: {before}s before to {after}s after the event; frames per event "
              f"{n_full:.1f} -> {tf.masks.sum(1).mean():.1f} (min {tf.masks.sum(1).min()})")
    if not args.keep_all_labels:
        tf = tf.drop_labels(dropped_labels(cfg)).merge_labels(label_merges(cfg))
    split = split_by_game(tf.games, tf.labels, load_manifest(repo_path(cfg, "manifest")),
                          args.val_games, args.test_games)
    classes = sorted(set(tf.labels[split.train]))
    cls_idx = {c: i for i, c in enumerate(classes)}
    keep = split.train | split.val | split.test
    y = np.array([cls_idx.get(label, -1) for label in tf.labels])
    print(f"Events: train={split.train.sum()} val={split.val.sum()} test={split.test.sum()}  "
          f"classes={len(classes)}  dim={tf.features.shape[2]}")

    masks = tf.masks.astype(np.float32)
    train_frames = tf.features[split.train][tf.masks[split.train]]
    mu, sd = train_frames.mean(0), train_frames.std(0) + 1e-6
    feats = ((tf.features - mu) / sd) * masks[:, :, None]  # keep padding at zero
    X, M, Y = torch.tensor(feats), torch.tensor(masks), torch.tensor(y, dtype=torch.long)

    def loader(mask, shuffle, generator=None):
        ds = torch.utils.data.TensorDataset(X[mask & keep], M[mask & keep], Y[mask & keep])
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, generator=generator)

    def predict_proba(model, mask):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(xb.to(device), mb.to(device)).softmax(1).cpu()
                              for xb, mb, _ in loader(mask, False)]).numpy()

    def predict(model, mask):
        return predict_proba(model, mask).argmax(1)

    def train(cls, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = cls(tf.features.shape[2], args.hidden_dim, len(classes), tf.features.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        best_val, best_state, wait, stopped = (-1.0, -1.0), None, 0, 0
        y_val = y[split.val & keep]
        # One generator per run, so each epoch gets a new (but reproducible) order.
        order = torch.Generator().manual_seed(seed)
        for epoch in range(1, args.epochs + 1):
            stopped = epoch
            model.train()
            for xb, mb, yb in loader(split.train, True, order):
                opt.zero_grad()
                crit(model(xb.to(device), mb.to(device)), yb.to(device)).backward()
                opt.step()
            pred = predict(model, split.val)
            val = (f1_score(y_val, pred, average="macro", zero_division=0), float(np.mean(pred == y_val)))
            if val > best_val:  # macro-F1, then accuracy
                best_val, wait = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= args.patience:
                    break
        model.load_state_dict(best_state)
        return model, best_val, stopped

    y_test, y_train = y[split.test & keep], y[split.train & keep]
    test_games = tf.games[split.test & keep]
    rows, runs = [], {}
    for name in args.models:
        cls = MODELS[name]
        for seed in args.seeds:
            model, (val_f1, val_acc), stopped = train(cls, seed)
            runs[(name, seed)] = model
            rows.append({"model": name, "seed": seed, "val_macro_f1": val_f1, "val_acc": val_acc,
                         "stopped_epoch": stopped})
            print(f"{name:14s} seed {seed}  stopped@{stopped:3d}  best val macro-F1={val_f1:.3f} "
                  f"accuracy={val_acc:.1%}")
    df = pd.DataFrame(rows)
    by_model = (df.groupby("model").agg(mean=("val_macro_f1", "mean"), std=("val_macro_f1", "std"),
                                        val_acc=("val_acc", "mean"))
                .sort_values(["mean", "val_acc"], ascending=False))
    chosen = by_model.index[0]
    if args.val_only:
        df.to_csv(out_dir / "sequence_results.csv", index=False)
        summary = (f"Validation only (no test scores). Best: {chosen}, mean val macro-F1 "
                   f"{by_model.loc[chosen, 'mean']:.4f} over {len(args.seeds)} seed(s)\n\n"
                   f"{by_model.to_string()}\n")
        (out_dir / "sequence_validation_report.txt").write_text(summary)
        print("\n" + summary)
        return

    # Test only the chosen model, once per seed.
    test_rows, preds, test_probs = [], [], []
    for seed in args.seeds:
        probs = predict_proba(runs[(chosen, seed)], split.test)
        pred = probs.argmax(1)
        preds.append(pred)
        test_probs.append(probs)
        test_rows.append({"seed": seed, **classification_scores(y_test, pred, y_train)})
    test_df = pd.DataFrame(test_rows)
    in_test = split.test & keep
    pd.concat([prediction_rows(test_games, tf.halves[in_test], tf.event_tags[in_test],
                             [classes[i] for i in y_test], [classes[i] for i in pred], seed=seed,
                             probs=probs, classes=classes)
               for seed, pred, probs in zip(args.seeds, preds, test_probs, strict=True)]).to_csv(
        out_dir / "sequence_test_predictions.csv", index=False)
    # Validation predictions of the chosen model, so thresholds on confidence (e.g. when commentary
    # should hedge) can be picked without looking at the test set.
    in_val = split.val & keep
    val_rows = []
    for seed in args.seeds:
        probs = predict_proba(runs[(chosen, seed)], split.val)
        val_rows.append(prediction_rows(tf.games[in_val], tf.halves[in_val], tf.event_tags[in_val],
                                        tf.labels[in_val], [classes[i] for i in probs.argmax(1)],
                                        seed=seed, probs=probs, classes=classes))
    pd.concat(val_rows).to_csv(out_dir / "sequence_val_predictions.csv", index=False)
    ci = bootstrap_ci(np.tile(y_test, len(preds)), np.concatenate(preds), np.tile(test_games, len(preds)),
                      seed=args.seeds[0])

    df.to_csv(out_dir / "sequence_results.csv", index=False)
    test_df.to_csv(out_dir / "sequence_test_by_seed.csv", index=False)
    acc, f1, bal = test_df["Accuracy"], test_df["Macro_F1"], test_df["Balanced_accuracy"]
    spread = (lambda v: f" +/- {v.std(ddof=1):.4f} (sd over {len(v)} seeds)") if len(args.seeds) > 1 else (lambda v: "")
    summary = (f"Chosen on validation: {chosen} (mean val macro-F1 {by_model.loc[chosen, 'mean']:.4f} "
               f"over {len(args.seeds)} seed(s))\n"
               f"Test games: {' '.join(split.test_games)}  ({len(y_test)} events)\n"
               f"Test accuracy {acc.mean():.4f}{spread(acc)}  macro-F1 {f1.mean():.4f}{spread(f1)}  "
               f"balanced accuracy {bal.mean():.4f}{spread(bal)}  "
               f"majority-class baseline {test_df['Majority_baseline'].iloc[0]:.4f}\n"
               f"95% interval, resampling test games (all seeds pooled): accuracy "
               f"{ci['Accuracy_CI'][0]:.3f}-{ci['Accuracy_CI'][1]:.3f}, macro-F1 "
               f"{ci['Macro_F1_CI'][0]:.3f}-{ci['Macro_F1_CI'][1]:.3f}\n")
    present = sorted(set(y_test) | set(preds[0]))
    report = classification_report(y_test, preds[0], labels=present,
                                   target_names=[classes[i] for i in present], zero_division=0)
    details = (f"\nValidation macro-F1 by model (mean, sd over seeds; val_acc is mean accuracy):\n{by_model.to_string()}\n\n"
               f"Per-class report (seed {args.seeds[0]}):\n{report}")
    (out_dir / "sequence_classification_report.txt").write_text(summary + details)
    print("\n" + summary + details)


if __name__ == "__main__":
    main()
