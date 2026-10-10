#!/bin/bash
cd /home/ivenwu/projects/MFTG
. /home/ivenwu/projects/MFTG/colab/env.sh; S=$SCRATCH
colab upload -s $SES mftg/train_mftg.py /content/code/train_mftg.py 2>&1 | grep -i fail
for p in build/mftg/upload_flow/flow.tgz.part*; do
  for try in 1 2 3 4 5; do
    if colab upload -s $SES $p /content/up/$(basename $p) 2>&1 | grep -q Uploaded; then echo "ok $(basename $p) $(date +%T)"; break; fi
    echo "retry $try $(basename $p)"; sleep 5
  done
done
echo done > $S/FLOW_DONE; colab upload -s $SES $S/FLOW_DONE /content/up/FLOW_DONE 2>&1 | grep -i upload; echo FLOW_UPLOAD_DONE
