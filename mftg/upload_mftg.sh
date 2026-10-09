#!/bin/bash
cd /home/ivenwu/projects/MFTG
mkdir -p ${MFTG_SCRATCH:-/tmp/mftg_colab}

for f in mftg/train_mftg.py; do colab upload -s ${SES:-e2e1} $f /content/code/$(basename $f) 2>&1 | grep -i "fail" ; done
for p in build/mftg/upload_e2e/e2e.tgz.part*; do
  for try in 1 2 3 4 5; do
    if colab upload -s ${SES:-e2e1} $p /content/up/$(basename $p) 2>&1 | grep -q Uploaded; then echo "ok $(basename $p) $(date +%T)"; break; fi
    echo "retry $try $(basename $p)"; sleep 5
  done
done
echo UPLOAD_DONE
echo done > ${MFTG_SCRATCH:-/tmp/mftg_colab}/UPLOAD_DONE; colab upload -s ${SES:-e2e1} ${MFTG_SCRATCH:-/tmp/mftg_colab}/UPLOAD_DONE /content/up/UPLOAD_DONE 2>&1 | grep -i upload
