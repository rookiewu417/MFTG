#!/bin/bash
# usage: submit.sh ORDER TAG LANE "ARGS" [OUTDIR]   -> queues a job for the in-kernel driver
. /home/ivenwu/projects/ml-factor/v2/colab/env.sh
f=/tmp/claude-1000/-home-ivenwu-projects-ml-factor/31aa1f34-2077-4b46-bf21-6dddcfa4c5b1/scratchpad/$1_$2.json
printf '{"tag": "%s", "lane": %s, "args": "%s", "out": "%s", "script": "%s"}\n' "$2" "$3" "$4" "${5:-$2}" "${6:-train.py}" > $f
colab upload -s $SES $f /content/queue/$1_$2.json 2>&1 | grep -q Uploaded && echo "queued $1_$2" || echo "FAILED $1_$2"
