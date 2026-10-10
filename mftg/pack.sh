#!/bin/bash
# Rebuild the Colab upload bundles from the symlink packs:
#   build/mftg/e2e_pack/e2e       (daily + intraday)  -> build/mftg/upload_e2e/e2e.tgz.partNNN
#   build/mftg/e2e_daily_pack/e2e (daily only)        -> build/mftg/upload_e2e_daily/e2e.tgz.partNNN
# Parts are 60 MiB; driver_mftg.py unpacks them into /content/e2e.
set -euo pipefail
cd /home/ivenwu/projects/MFTG/build/mftg
for k in ${@:-e2e e2e_daily}; do
  out=upload_$k; tmp=$out.tmp
  rm -rf $tmp && mkdir $tmp
  tar -czhf - -C ${k}_pack e2e | split -b 60M -d -a 3 - $tmp/e2e.tgz.part
  rm -rf $out && mv $tmp $out
  echo "$out: $(ls $out | wc -l) parts, $(du -sh $out | cut -f1)"
done
