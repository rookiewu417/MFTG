#!/bin/bash
# pull every finished *_pred.npz / *_log.json not yet present locally (contents API only; kernel stays busy)
. /home/ivenwu/projects/ml-factor/v2/colab/env.sh
R=/home/ivenwu/projects/ml-factor/v2/runs
lsd() { colab ls -s $SES "$1" 2>/dev/null | grep -v "^\[colab\]\|^$" | awk '{print $NF}' | sed 's#/$##'; }
for run in $(lsd /content/out); do
  case "$run" in *.log|*.json|*.tmp) continue;; esac
  for m in $(lsd /content/out/$run); do
    for f in $(lsd /content/out/$run/$m | grep -E "_pred.npz|_log.json"); do
      [ -f $R/$run/$m/$f ] && continue
      mkdir -p $R/$run/$m; colab download -s $SES /content/out/$run/$m/$f $R/$run/$m/$f >/dev/null 2>&1 && echo "got $run/$m/$f"
    done
  done
done
