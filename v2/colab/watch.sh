#!/bin/bash
# usage: watch.sh TAG... -> prints each new result/error line once; exits when the driver queue is empty and idle
cd /home/ivenwu/projects/ml-factor/v2
S=/tmp/claude-1000/-home-ivenwu-projects-ml-factor/31aa1f34-2077-4b46-bf21-6dddcfa4c5b1/scratchpad
SEEN=$S/seen_lines.txt; touch $SEEN
while true; do
  colab/status.sh "$@" > $S/w_raw.txt 2>&1
  [ -s $S/w_raw.txt ] || echo "STATUS UNREACHABLE $(date +%T)"
  awk '/^## /{t=$2} /==>|Error|Traceback/{print t": "$0}' $S/w_raw.txt | sort -u > $S/w_cur.txt
  comm -13 <(sort -u $SEEN) $S/w_cur.txt
  sort -u $SEEN $S/w_cur.txt -o $SEEN
  if grep -q '"queued": \[\]' $S/w_raw.txt && grep -q '"running": {}' $S/w_raw.txt; then echo ALL_DONE; exit 0; fi
  sleep 240
done
