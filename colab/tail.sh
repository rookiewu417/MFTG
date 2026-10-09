#!/bin/bash
# usage: [SES=<session>] tail.sh TAG [n]
. /home/ivenwu/projects/MFTG/colab/env.sh
echo "import subprocess; print(subprocess.run('tail -n ${2:-20} /content/out/$1.log', shell=True, capture_output=True, text=True).stdout)" | timeout 120 colab exec -s $SES 2>&1 | grep -v -i "new version\|colab update\|silence"
