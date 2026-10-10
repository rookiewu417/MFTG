#!/bin/bash
# usage: submit_jobs.sh SES   -> queues every row of build/keeper/jobs_<SES>.tsv (tag, lane, args, out, script, prio) via submit.sh
. /home/ivenwu/projects/MFTG/colab/env.sh
SES=${1:-$SES}; J=/home/ivenwu/projects/MFTG/build/keeper/jobs_$SES.tsv
i=0
while IFS=$'\t' read -r tag lane args out script prio; do
  [ -z "$tag" ] && continue
  SES=$SES /home/ivenwu/projects/MFTG/colab/submit.sh $(printf '%02d' $i) "$tag" "$lane" "$args" "$out" "$script"; i=$((i+1))
done < $J
