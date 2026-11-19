#!/bin/bash
# Step 3: window / look-ahead test. BiLSTM, merged labels, 3 seeds, validation only (no test scores).
cd ~/soccer
export SOCCER_DATA_ROOT=~/soccer-data SOCCER_MANIFEST=configs/games_epl_1720.csv PYTHONPATH=.
FEATS=~/soccer-data/runs/epl1720_clf/dino/features/temporal_dinov2-base
OUT=~/soccer-data/runs/epl1720_clf/dino/exp_window
run() {  # name, extra args
  echo "=== $1 $(date)"
  .venv/bin/python scripts/sequence_classifier.py --features $FEATS --seeds 42 43 44 --models BiLSTM \
    --val-only --device cuda --out-dir $OUT/$1 "${@:2}" || echo "=== FAILED $1"
}
run full
run b15_a15 --before-sec 15 --after-sec 15
run b10_a10 --before-sec 10 --after-sec 10
run b30_a10 --before-sec 30 --after-sec 10
run b30_a5  --before-sec 30 --after-sec 5
run b30_a0  --before-sec 30 --after-sec 0
echo "=== ALL DONE $(date)"
