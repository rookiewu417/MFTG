#!/bin/bash
# keeps a websocket client attached to the busy kernel: each exec queues a no-op cell behind the driver and waits;
# when the wait times out or the socket drops, it reconnects immediately.
. /home/ivenwu/projects/ml-factor/v2/colab/env.sh
while colab sessions 2>/dev/null | grep -q "$SES"; do
  echo "pass" | timeout 3700 colab exec -s $SES --timeout 3600 >/dev/null 2>&1
  echo "keeper reconnect $(date +%T)"
  sleep 5
done
echo "keeper: session gone $(date +%T)"
