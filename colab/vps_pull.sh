#!/bin/bash
# usage: [LINE=mftg] colab/vps_pull.sh SES   -> download what vps_watch.sh put in Google Drive (MFTG/SES/) into runs/$LINE/,
# copy its logs/status from the VPS into build/keeper/SES/ and show the last log lines
. /home/ivenwu/projects/MFTG/colab/env.sh
SES=$1; [ -n "$SES" ] || { echo "usage: colab/vps_pull.sh SES"; exit 1; }
~/.local/share/uv/tools/google-colab-cli/bin/python3 $MFTG_ROOT/colab/drive_sync.py pull MFTG/$SES $MFTG_ROOT/runs/$LINE/
mkdir -p $K/$SES; rsync -a vps:mftg_colab/$SES/{watch.log,keeper.log,status.json,failed.txt,uploaded.txt} $K/$SES/ 2>/dev/null
tail -3 $K/$SES/watch.log
