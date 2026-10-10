#!/bin/bash
# usage: [ACCT=main|<name>] colab/start.sh SES GPU   (GPU: T4 | L4 | A100 | H100)
# Creates the session, records its account, starts the in-kernel driver (mftg/driver_mftg.py) and the local keeper and
# supervisor (logs in build/keeper/). Then upload data (mftg/upload_*.sh) and queue jobs (colab/submit_jobs.sh).
SES=$1; GPU=$2
[ -n "$SES" ] && [ -n "$GPU" ] || { echo "usage: [ACCT=..] colab/start.sh SES GPU"; exit 1; }
K=/home/ivenwu/projects/MFTG/build/keeper; mkdir -p $K
echo "${ACCT:-main}" > $K/account_$SES
export SES; . /home/ivenwu/projects/MFTG/colab/env.sh
echo "session $SES on account $ACCT, gpu $GPU"
colab new -s $SES --gpu $GPU || exit 1
cd $MFTG_ROOT
setsid nohup colab exec -s $SES -f mftg/driver_mftg.py --timeout 200000 > $K/driver_$SES.log 2>&1 < /dev/null &
sleep 30
SES=$SES setsid nohup colab/keeper.sh > $K/keeper_$SES.log 2>&1 < /dev/null &
setsid nohup colab/supervisor.sh $SES > $K/supervisor_$SES.log 2>&1 < /dev/null &
echo "driver, keeper, supervisor started for $SES"
