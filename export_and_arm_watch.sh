#!/usr/bin/env bash
set -euo pipefail

# Credential-safe export watcher. It uses an already authenticated SSH control
# socket; no password is stored in this file, its environment, or its logs.
ROOT="$(cd "$(dirname "$0")" && pwd)"
SOCK="${SSH_CONTROL_SOCKET:-/tmp/repro_factor_ssh_45080.sock}"
HOST="${REMOTE_HOST:-connect.weste.seetacloud.com}"
PORT="${REMOTE_PORT:-45080}"
USER_NAME="${REMOTE_USER:-root}"
REMOTE="${USER_NAME}@${HOST}"
REMOTE_ROOT="/root/autodl-tmp/quant/repro_ml_factor_luna_20260917"
OUT="$ROOT/local_exports"
LOG="$OUT/export_and_arm_watch.log"
INTERVAL="${EXPORT_WATCH_INTERVAL:-60}"
mkdir -p "$OUT"
log(){ printf '%s %s\n' "$(date '+%F %T %z')" "$*" >> "$LOG"; }
remote(){ ssh -S "$SOCK" -p "$PORT" -o BatchMode=yes "$REMOTE" "$@"; }

if ! ssh -S "$SOCK" -p "$PORT" -o BatchMode=yes "$REMOTE" true >/dev/null 2>&1; then
  log "ERROR control socket unavailable: $SOCK"
  exit 2
fi

while :; do
  run_id="$(remote "cat '$REMOTE_ROOT/finalization/GUARD_RUN_ID'" 2>/dev/null || true)"
  if [[ ! "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then log "WAIT invalid_or_missing_run_id"; sleep "$INTERVAL"; continue; fi
  if ! remote "test -s '$REMOTE_ROOT/finalization/SERVER_EXPORT_READY'" >/dev/null 2>&1; then
    log "WAIT run_id=$run_id server_export_not_ready"; sleep "$INTERVAL"; continue
  fi

  bundle_sha="$(remote "awk '{print \$1}' '$REMOTE_ROOT/finalization/RESULT_BUNDLE.sha256'" 2>/dev/null || true)"
  manifest_sha="$(remote "sha256sum '$REMOTE_ROOT/finalization/finalization_manifest.json' | awk '{print \$1}'" 2>/dev/null || true)"
  if [[ ! "$bundle_sha" =~ ^[0-9a-f]{64}$ ]] || [[ ! "$manifest_sha" =~ ^[0-9a-f]{64}$ ]]; then
    log "WAIT run_id=$run_id invalid_server_hashes"; sleep "$INTERVAL"; continue
  fi
  dest="$OUT/RESULT_BUNDLE_${run_id}.tar.gz"
  tmp="$dest.tmp"
  rm -f "$tmp"
  if ! scp -q -o ControlPath="$SOCK" -P "$PORT" -o BatchMode=yes "$REMOTE:$REMOTE_ROOT/finalization/RESULT_BUNDLE.tar.gz" "$tmp"; then
    log "EXPORT_FAIL run_id=$run_id scp_bundle"; rm -f "$tmp"; sleep "$INTERVAL"; continue
  fi
  local_sha="$(sha256sum "$tmp" | awk '{print $1}')"
  if [ "$local_sha" != "$bundle_sha" ]; then log "EXPORT_FAIL run_id=$run_id bundle_sha_mismatch"; rm -f "$tmp"; sleep "$INTERVAL"; continue; fi
  mv "$tmp" "$dest"
  scp -q -o ControlPath="$SOCK" -P "$PORT" -o BatchMode=yes "$REMOTE:$REMOTE_ROOT/finalization/finalization_manifest.json" "$OUT/finalization_manifest_${run_id}.json"
  local_manifest_sha="$(sha256sum "$OUT/finalization_manifest_${run_id}.json" | awk '{print $1}')"
  if [ "$local_manifest_sha" != "$manifest_sha" ]; then log "EXPORT_FAIL run_id=$run_id manifest_sha_mismatch"; sleep "$INTERVAL"; continue; fi
  printf 'run_id=%s\nbundle_sha256=%s\nmanifest_sha256=%s\n' "$run_id" "$bundle_sha" "$manifest_sha" > "$OUT/export_receipt_${run_id}.txt"
  remote "umask 077; printf 'run_id=%s\\nbundle_sha256=%s\\nmanifest_sha256=%s\\n' '$run_id' '$bundle_sha' '$manifest_sha' > '$REMOTE_ROOT/finalization/LOCAL_EXPORT_VERIFIED.tmp'; mv '$REMOTE_ROOT/finalization/LOCAL_EXPORT_VERIFIED.tmp' '$REMOTE_ROOT/finalization/LOCAL_EXPORT_VERIFIED'; printf 'run_id=%s\\nbundle_sha256=%s\\nmanifest_sha256=%s\\n' '$run_id' '$bundle_sha' '$manifest_sha' > '$REMOTE_ROOT/finalization/ARM_$run_id.tmp'; mv '$REMOTE_ROOT/finalization/ARM_$run_id.tmp' '$REMOTE_ROOT/finalization/ARM_$run_id'"
  log "EXPORT_VERIFIED_AND_ARMED run_id=$run_id bundle_sha256=$bundle_sha manifest_sha256=$manifest_sha"
  exit 0
done
