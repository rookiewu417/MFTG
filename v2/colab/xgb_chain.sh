#!/bin/bash
# waits for the running NN jobs to exit, then runs the remaining jobs one at a time on the free GPU
cd /content/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "train.py --out" >/dev/null; do sleep 30; done
python train.py --out /content/out/runA --models fusion --neutral 0 > /content/out/runA_fusion.log 2>&1
python train.py --out /content/out/runB --models xgb --neutral 1 > /content/out/runB_xgb.log 2>&1
python train.py --out /content/out/runA_xgb --models xgb --neutral 0 > /content/out/runA_xgb.log 2>&1
echo CHAIN_DONE >> /content/out/runA_xgb.log
