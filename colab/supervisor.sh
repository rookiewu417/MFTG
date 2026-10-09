#!/bin/bash
# usage: [LINE=mftg|repro] supervisor.sh SES -- local supervisor: every 5 min pull finished predictions (sync.sh), driver status and job logs
# into build/keeper/SES/; flags failed jobs; when the driver has nothing running or queued twice in a row: final sync + stop the VM.
export SES=$1; . /home/ivenwu/projects/MFTG/colab/env.sh
D=/home/ivenwu/projects/MFTG/build/keeper/$SES; mkdir -p $D; idle=0
while colab sessions 2>/dev/null | grep -q "\[$SES\]"; do
  bash /home/ivenwu/projects/MFTG/colab/sync.sh
  st=$D/status.json; colab download -s $SES /content/out/status.json $st >/dev/null 2>&1
  for t in $(tr -d '\n ' < $st 2>/dev/null | grep -o '"tag":"[^"]*"' | cut -d'"' -f4) $(tr -d '\n ' < $st 2>/dev/null | grep -o '"running":{[^}]*}' | grep -o ':"[^"]*"' | tr -d ':"'); do
    colab download -s $SES /content/out/$t.log $D/$t.log >/dev/null 2>&1
  done
  tr -d '\n ' < $st 2>/dev/null | grep -o '"tag":"[^"]*","lane":[0-9]*,"rc":-\?[1-9][0-9]*' | sed "s/^/$(date '+%F %T') FAILED /" | sort -u >> $D/failed.txt
  if [ -f $st ] && grep -q '"data": true' $st && grep -q '"running": {}' $st && grep -q '"queued": \[\]' $st; then idle=$((idle+1)); else idle=0; fi
  echo "supervisor $SES $(date '+%F %T') idle=$idle $(tr -d '\n ' < $st 2>/dev/null | grep -o '"running":{[^}]*}')"
  if [ $idle -ge 2 ]; then
    bash /home/ivenwu/projects/MFTG/colab/sync.sh; echo "all jobs done on $SES -> stopping $(date '+%F %T')"
    colab stop -s $SES; break
  fi
  sleep 300
done
echo "supervisor $SES exit $(date '+%F %T')"
