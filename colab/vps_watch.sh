#!/bin/bash
# Low-footprint keeper + result fetcher for a Colab session, meant for a tiny always-on VPS (1 core / 1 GB):
#   usage (on the VPS): setsid nohup nice -n 19 ~/mftg_colab/vps_watch.sh SES > ~/mftg_colab/SES/watch.log 2>&1 < /dev/null &
#   needs ~/mftg_colab/{keeper_exec.py, drive_sync.py, SES/manifest.tsv} and the colab CLI logged in (~/.config/colab-cli).
# manifest.tsv: tag <TAB> remote dir under /content/out <TAB> comma-separated blocks (written by colab/vps_push.sh).
# - keeper: one patched `colab exec` (keeper_exec.py, no busy-wait) queued behind the driver, re-attached hourly
# - every 10 min: one `colab sessions` + one status.json download; predictions of a job are downloaded only once
#   the driver reports it finished (rc 0), each file goes straight to Google Drive (MFTG/SES/<remote dir>/, drive_sync.py)
#   and is deleted on the VPS; when nothing is running or queued: final fetch, `colab stop`, exit.
# Pull the results with colab/vps_pull.sh (from Drive).
SES=$1; W=~/mftg_colab; D=$W/$SES; mkdir -p $D; touch $D/fetched.txt $D/uploaded.txt
C=~/.local/bin/colab; PY=~/.local/share/uv/tools/google-colab-cli/bin/python3
alive() { $C sessions 2>/dev/null | grep -q "\[$SES\]"; }
log() { echo "$(date '+%F %T') $*"; }

( while alive; do
    echo pass | timeout 3700 $PY $W/keeper_exec.py exec -s $SES --timeout 7200 >/dev/null 2>&1
    log "keeper reconnect"; sleep 5
  done; log "keeper: session gone" ) >> $D/keeper.log 2>&1 &

fetch() {   # tag -> 0 once every file of the job is in Drive (one file on the VPS disk at a time)
  local ok=0
  while IFS=$'\t' read -r tag dir blocks; do
    [ "$tag" = "$1" ] || continue
    for b in ${blocks//,/ }; do
      for f in ${b}_pred.npz ${b}_log.json; do
        grep -qx "$dir/$f" $D/uploaded.txt && continue
        mkdir -p $D/tmp; rm -f $D/tmp/$f
        if $C download -s $SES /content/out/$dir/$f $D/tmp/$f >/dev/null 2>&1 < /dev/null && [ -s $D/tmp/$f ] \
           && $PY $W/drive_sync.py up $D/tmp/$f MFTG/$SES/$dir < /dev/null; then echo "$dir/$f" >> $D/uploaded.txt; log "drive $dir/$f"; else ok=1; fi
        rm -f $D/tmp/$f
      done
    done
  done < $D/manifest.tsv
  return $ok
}

fetch_done() {
  for t in $(tr -d '\n ' < $D/status.json | grep -o '"tag":"[^"]*","lane":[0-9]*,"rc":0' | cut -d'"' -f4 | sort -u); do
    grep -qx "$t" $D/fetched.txt 2>/dev/null && continue
    fetch $t && echo "$t" >> $D/fetched.txt
  done
}

while alive; do
  if $C download -s $SES /content/out/status.json $D/status.json.new >/dev/null 2>&1; then
    mv $D/status.json.new $D/status.json
    st=$(tr -d '\n ' < $D/status.json)
    echo "$st" | grep -o '"tag":"[^"]*","lane":[0-9]*,"rc":-\?[1-9][0-9]*' | sort -u > $D/failed.txt
    fetch_done
    log "running $(echo "$st" | grep -o '"running":{[^}]*}') fetched $(wc -l < $D/fetched.txt 2>/dev/null || echo 0)/$(wc -l < $D/manifest.tsv)"
    if echo "$st" | grep -q '"data":true' && echo "$st" | grep -q '"running":{}' && echo "$st" | grep -q '"queued":\[\]' \
       && ! echo "$st" | grep -q '"done":\[\]'; then
      fetch_done; log "all jobs done -> stopping $SES"; $C stop -s $SES; break
    fi
  fi
  sleep 600
done
log "watch $SES exit"
