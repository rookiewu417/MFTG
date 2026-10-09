#!/bin/bash
# usage: SES=<session> mftg/upload_parts.sh "<local part glob>" <DONE marker name>
cd /home/ivenwu/projects/MFTG
. /home/ivenwu/projects/MFTG/colab/env.sh; S=$SCRATCH
colab upload -s $SES mftg/train_mftg.py /content/code/train_mftg.py 2>&1 | grep -i fail
for p in $1; do
  for try in 1 2 3 4 5 6; do
    if colab upload -s $SES $p /content/up/$(basename $p) 2>&1 | grep -q Uploaded; then echo "ok $(basename $p) $(date +%T)"; break; fi
    echo "retry $try $(basename $p)"; sleep 5
  done
done
echo done > $S/$2; colab upload -s $SES $S/$2 /content/up/$2 2>&1 | grep -i upload; echo "$2 SENT"
