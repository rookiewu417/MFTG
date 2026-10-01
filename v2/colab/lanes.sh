#!/bin/bash
# two GPU lanes: tree models and neural fusion models, run sequentially within each lane
cd /content/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
( python train.py --out /content/out/runA_xgb --models xgb --neutral 0 > /content/out/runA_xgb.log 2>&1
  python train.py --out /content/out/runB --models xgb --neutral 1 > /content/out/runB_xgb.log 2>&1
  echo LANE1_DONE > /content/out/lane1.done ) &
( python train.py --out /content/out/runA --models fusion --neutral 0 > /content/out/runA_fusion.log 2>&1
  python train.py --out /content/out/runB --models fusion --neutral 1 > /content/out/runB_fusion.log 2>&1
  echo LANE2_DONE > /content/out/lane2.done ) &
wait
