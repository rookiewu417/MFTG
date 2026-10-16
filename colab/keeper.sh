#!/bin/bash
# keeps a websocket client attached to the busy kernel: each exec queues a no-op cell behind the driver and waits;
# when the wait times out or the socket drops, it reconnects immediately.
# keeper_exec.py instead of plain `colab exec`: the plain CLI busy-waits (one core at 100%) while its cell is queued.
# --timeout is longer than the `timeout` wrapper so the wait never reaches its deadline (past it the loop spins too).
. /home/ivenwu/projects/MFTG/colab/env.sh
PY=/home/ivenwu/.local/share/uv/tools/google-colab-cli/bin/python3
[ "$ACCT" != main ] && export HOME=/home/ivenwu/.colab-accounts/$ACCT
while colab sessions 2>/dev/null | grep -q "$SES"; do
  echo "pass" | timeout 3700 $PY $MFTG_ROOT/colab/keeper_exec.py exec -s $SES --timeout 7200 >/dev/null 2>&1
  echo "keeper reconnect $(date +%T)"
  sleep 5
done
echo "keeper: session gone $(date +%T)"
