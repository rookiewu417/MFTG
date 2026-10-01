#!/bin/bash
cd /home/ivenwu/projects/ml-factor/v2; . colab/env.sh
colab upload -s $SES colab/train.py /content/code/train.py 2>&1 | grep -i fail
for p in build/upload2/pack2.tgz.part*; do
  for try in 1 2 3 4 5; do
    if colab upload -s $SES $p /content/up/$(basename $p) 2>&1 | grep -q Uploaded; then echo "ok $(basename $p) $(date +%T)"; break; fi
    echo "retry $try $(basename $p)"; sleep 5
  done
done
A9="2022H2,2023H1,2023H2,2024H1,2024H2,2025H1,2025H2,2026H1,2026H2"
U="--data /content/data2 --unpack /content/up/pack2.tgz.part"
colab/submit.sh 38 p2_xgb_top_fast 2 "--models xgb --top_w 2 --xgb_lr 0.08 --xgb_rounds 6000 --blocks $A9 $U"
colab/submit.sh 39 p2_xgb_neu 1 "--models xgb --neutral 1 --blocks $A9 $U"
echo P2_SUBMITTED
