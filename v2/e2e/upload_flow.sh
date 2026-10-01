#!/bin/bash
cd /home/ivenwu/projects/ml-factor/v2
S=/tmp/claude-1000/-home-ivenwu-projects-ml-factor/31aa1f34-2077-4b46-bf21-6dddcfa4c5b1/scratchpad
colab upload -s e2e1 e2e/train_e2e.py /content/code/train_e2e.py 2>&1 | grep -i fail
for p in build/upload_flow/flow.tgz.part*; do
  for try in 1 2 3 4 5; do
    if colab upload -s e2e1 $p /content/up/$(basename $p) 2>&1 | grep -q Uploaded; then echo "ok $(basename $p) $(date +%T)"; break; fi
    echo "retry $try $(basename $p)"; sleep 5
  done
done
echo done > $S/FLOW_DONE; colab upload -s e2e1 $S/FLOW_DONE /content/up/FLOW_DONE 2>&1 | grep -i upload; echo FLOW_UPLOAD_DONE
