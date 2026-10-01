#!/bin/bash
# usage: fetch.sh TAG...  -> downloads /content/out/<TAG> (+log) into v2/runs/
set -e
mkdir -p /home/ivenwu/projects/ml-factor/v2/runs
for t in "$@"; do
  echo "import subprocess; subprocess.run('cd /content/out && tar -czf /content/$t.tgz $t $t.log', shell=True)" | timeout 300 colab exec -s mlf2 >/dev/null 2>&1
  timeout 900 colab download -s mlf2 /content/$t.tgz /home/ivenwu/projects/ml-factor/v2/runs/$t.tgz 2>&1 | grep -i "download"
  tar -xzf /home/ivenwu/projects/ml-factor/v2/runs/$t.tgz -C /home/ivenwu/projects/ml-factor/v2/runs && rm /home/ivenwu/projects/ml-factor/v2/runs/$t.tgz
done
