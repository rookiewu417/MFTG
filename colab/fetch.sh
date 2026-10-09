#!/bin/bash
# usage: [SES=<session>] [LINE=mftg|repro] fetch.sh TAG...  -> downloads /content/out/<TAG> (+ <TAG>.log) into runs/$LINE/
set -e
. /home/ivenwu/projects/MFTG/colab/env.sh
D=$MFTG_ROOT/runs/$LINE; mkdir -p "$D"
for t in "$@"; do
  echo "import subprocess; subprocess.run('cd /content/out && tar -czf /content/$t.tgz $t $t.log', shell=True)" | timeout 300 colab exec -s $SES >/dev/null 2>&1
  timeout 900 colab download -s $SES /content/$t.tgz $D/$t.tgz 2>&1 | grep -i "download"
  tar -xzf $D/$t.tgz -C $D && rm $D/$t.tgz
done
