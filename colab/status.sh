#!/bin/bash
# usage: status.sh [TAG...]  -> driver status plus the result lines of the given job logs
. /home/ivenwu/projects/MFTG/colab/env.sh
S=$SCRATCH/st; mkdir -p $S
colab download -s $SES /content/out/status.json $S/status.json >/dev/null 2>&1 && cat $S/status.json
for t in "$@"; do colab download -s $SES /content/out/$t.log $S/$t.log >/dev/null 2>&1 && { echo "## $t"; tr -d '\000' < $S/$t.log | grep -aE "==>|Error|Traceback" | tail -8; }; done
