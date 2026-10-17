#!/bin/bash
# usage: colab/vps_push.sh SES   -> hand a running Colab session over to the VPS (ssh host "vps", see ~/.ssh/config):
# copies the colab CLI login + session state, keeper_exec.py, vps_watch.sh, drive_sync.py and a manifest built from
# build/keeper/jobs_SES.tsv, then starts vps_watch.sh there (niced). Local keeper/supervisor can keep running or be closed.
set -e
SES=$1; [ -n "$SES" ] || { echo "usage: colab/vps_push.sh SES"; exit 1; }
. /home/ivenwu/projects/MFTG/colab/env.sh
[ "$ACCT" = main ] && CFG=~/.config/colab-cli || CFG=~/.colab-accounts/$ACCT/.config/colab-cli
M=$SCRATCH/manifest_$SES.tsv
# tag, remote dir (<out>/<--name>), blocks
awk -F'\t' '{n=""; b=""; k=split($3, a, " "); for (i=1;i<k;i++) {if (a[i]=="--name") n=a[i+1]; if (a[i]=="--blocks") b=a[i+1]}
             print $1 "\t" ($4==""?$1:$4) "/" n "\t" b}' $K/jobs_$SES.tsv > $M
ssh vps "mkdir -p ~/mftg_colab/$SES ~/.config/colab-cli"
scp -q $CFG/token.json $CFG/sessions.json $CFG/settings.json vps:.config/colab-cli/
# upload under a temporary name and rename: running watchers keep reading their own (old) copy of the script
scp -q $MFTG_ROOT/colab/keeper_exec.py $MFTG_ROOT/colab/drive_sync.py vps:mftg_colab/
scp -q $MFTG_ROOT/colab/vps_watch.sh vps:mftg_colab/vps_watch.sh.new && ssh vps "mv ~/mftg_colab/vps_watch.sh.new ~/mftg_colab/vps_watch.sh"
scp -q $M vps:mftg_colab/$SES/manifest.tsv
ssh vps "chmod +x ~/mftg_colab/vps_watch.sh; setsid nohup nice -n 19 ~/mftg_colab/vps_watch.sh $SES >> ~/mftg_colab/$SES/watch.log 2>&1 < /dev/null & echo started"
