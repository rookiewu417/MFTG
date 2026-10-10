#!/bin/bash
# usage: monitor.sh SES... -- local monitor, every 10 min. Relies on supervisor.sh (syncs build/keeper/<SES>/status.json + job logs).
# Jobs per session come from build/keeper/jobs_<SES>.tsv (tag, lane, args, out, script, critical|normal).
# Writes build/keeper/STATUS.md; appends ALERT lines to build/keeper/alerts.log for: session gone before its supervisor
# finished, failed jobs (critical ones re-queued as <tag>_r1/_r2, max 2), running job whose log has not grown for 60 min.
K=/home/ivenwu/projects/MFTG/build/keeper; SESS="$*"
acct() { cat $K/account_$1 2>/dev/null || echo main; }
colab_as() { local a=$(acct $1); shift; if [ "$a" = main ]; then colab "$@"; else COLAB_ACCT=$a /home/ivenwu/projects/MFTG/colab/acct/colab "$@"; fi; }
alert() { echo "$(date '+%F %T') ALERT $*" >> $K/alerts.log; }
note() { echo "$(date '+%F %T') $*" >> $K/alerts.log; }
note "monitor start: $SESS"
while [ -n "$SESS" ]; do
  live=$(for x in $SESS; do colab_as $x sessions; done 2>/dev/null); next=""      # each session listed under its own account
  {
  echo "# Colab 训练状态 $(date '+%F %T')"; echo
  for SES in $SESS; do
    L=$K/$SES; J=$K/jobs_$SES.tsv; R=$K/retries_$SES.txt; touch $R; st=$L/status.json
    if ! echo "$live" | grep -q "\[$SES\]"; then
      if grep -q "all jobs done on $SES" $K/supervisor_$SES.log 2>/dev/null; then note "$SES finished and stopped normally"; else alert "$SES session gone before all jobs finished"; fi
      echo "## $SES: 已结束"; echo; continue
    fi
    next="$next $SES"
    flat=$(tr -d '\n ' < $st 2>/dev/null); running=$(echo "$flat" | grep -o '"running":{[^}]*}')
    echo "## $SES (uptime $(echo "$flat" | grep -o '"uptime_min":[0-9.]*' | cut -d: -f2) min)"; echo
    echo "| 任务 | 状态 | 完成窗口 | 日志多久没更新 | 最新一行 |"; echo "|---|---|---|---|---|"
    while IFS=$'\t' read -r tag lane args out script prio; do
      f=$L/$tag.log; rounds=0; age=-; last=""
      if [ -f $f ]; then
        rounds=$(tr -d '\000' < $f | grep -ac "==> .* test rankIC"); size=$(stat -c %s $f); now=$(date +%s)
        last=$(tr -d '\000' < $f | grep -a "seed\|==>" | tail -n 1 | cut -c1-90 | tr '|' '/')
        prev=$(grep "^$tag " $K/sizes_$SES.txt 2>/dev/null | tail -n 1)
        if [ -z "$prev" ] || [ "$(echo $prev | cut -d' ' -f2)" != "$size" ]; then echo "$tag $size $now" >> $K/sizes_$SES.txt; changed=$now; else changed=$(echo $prev | cut -d' ' -f3); fi
        age=$(( (now - changed) / 60 ))
      fi
      state=排队; echo "$running" | grep -q "\"$tag\"" && state=运行中
      rc=$(echo "$flat" | grep -o "{\"tag\":\"$tag\",\"lane\":[0-9]*,\"rc\":-\?[0-9]*" | grep -o 'rc":-\?[0-9]*' | tail -n 1 | cut -d: -f2)
      if [ -n "$rc" ]; then if [ "$rc" = 0 ]; then state=完成; else state="失败 rc=$rc"; fi; fi
      echo "| $tag | $state | $rounds | ${age}m | $last |"
      if [ -n "$rc" ] && [ "$rc" != 0 ] && ! grep -qx "$tag" $R; then
        echo "$tag" >> $R
        err=$(tr -d '\000' < $f 2>/dev/null | grep -a "Error" | tail -n 1 | cut -c1-160)
        base=${tag%%_r[0-9]*}; n=$(grep -c "^$base" $R)
        if [ "$prio" = critical ] && [ $n -le 2 ]; then
          new="${base}_r$n"; tmp=$K/$new.json
          printf '{"tag": "%s", "lane": %s, "args": "%s", "out": "%s", "script": "%s"}\n' "$new" "$lane" "$args" "$out" "$script" > $tmp
          if colab_as $SES upload -s $SES $tmp /content/queue/90_$new.json 2>&1 | grep -q Uploaded; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$new" "$lane" "$args" "$out" "$script" "$prio" >> $J
            alert "$SES $tag failed ($err) -> resubmitted as $new on lane $lane"
          else alert "$SES $tag failed ($err) -> resubmit upload FAILED"; fi
        else alert "$SES $tag failed ($err) -> not retried"; fi
      fi
      if [ "$state" = 运行中 ] && [ "$age" != - ] && [ "$age" -gt 60 ]; then alert "$SES $tag log unchanged for ${age} min (stalled?)"; fi
    done < $J
    echo
  done
  echo "本地预测文件: $(find /home/ivenwu/projects/MFTG/runs/mftg -newer $K/.round_start -name '*_pred.npz' | wc -l) 个（本轮新增）"
  for x in $(echo $SESS | tr ' ' '\n' | sort -u); do echo "$x [$(acct $x)]: $(timeout 60 colab_as $x usage 2>/dev/null | head -n 2 | tr '\n' ' ')"; echo; done
  } > $K/STATUS.md.tmp && mv $K/STATUS.md.tmp $K/STATUS.md
  SESS=$(echo $next)
  [ -n "$SESS" ] && sleep 600
done
note "monitor exit: all sessions ended"
