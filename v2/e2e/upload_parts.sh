#!/bin/bash
# usage: SES=<session> e2e/upload_parts.sh "<local part glob>" <DONE marker name>
cd /home/ivenwu/projects/ml-factor/v2
S=/tmp/claude-1000/-home-ivenwu-projects-ml-factor/31aa1f34-2077-4b46-bf21-6dddcfa4c5b1/scratchpad
colab upload -s $SES e2e/train_e2e.py /content/code/train_e2e.py 2>&1 | grep -i fail
for p in $1; do
  for try in 1 2 3 4 5 6; do
    if colab upload -s $SES $p /content/up/$(basename $p) 2>&1 | grep -q Uploaded; then echo "ok $(basename $p) $(date +%T)"; break; fi
    echo "retry $try $(basename $p)"; sleep 5
  done
done
echo done > $S/$2; colab upload -s $SES $S/$2 /content/up/$2 2>&1 | grep -i upload; echo "$2 SENT"
