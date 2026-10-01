#!/bin/bash
cd /home/ivenwu/projects/ml-factor/v2

for f in e2e/train_e2e.py; do colab upload -s e2e1 $f /content/code/$(basename $f) 2>&1 | grep -i "fail" ; done
for p in build/upload_e2e/e2e.tgz.part*; do
  for try in 1 2 3 4 5; do
    if colab upload -s e2e1 $p /content/up/$(basename $p) 2>&1 | grep -q Uploaded; then echo "ok $(basename $p) $(date +%T)"; break; fi
    echo "retry $try $(basename $p)"; sleep 5
  done
done
echo UPLOAD_DONE
echo done > /tmp/claude-1000/-home-ivenwu-projects-ml-factor/31aa1f34-2077-4b46-bf21-6dddcfa4c5b1/scratchpad/UPLOAD_DONE; colab upload -s e2e1 /tmp/claude-1000/-home-ivenwu-projects-ml-factor/31aa1f34-2077-4b46-bf21-6dddcfa4c5b1/scratchpad/UPLOAD_DONE /content/up/UPLOAD_DONE 2>&1 | grep -i upload
