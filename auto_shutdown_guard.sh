#!/usr/bin/env bash
set -u

# Automatic shutdown was explicitly revoked by the user. This file is retained
# only as a diagnostic gate and must never stop the server.
ROOT="$(cd "${1:-.}" && pwd)"
LOG="$ROOT/auto_shutdown_guard.log"
OUT="$ROOT/finalization"
INTERVAL="${GUARD_INTERVAL:-60}"
MODE="${2:-watch}"
mkdir -p "$OUT"
log() { printf '%s %s\n' "$(date '+%F %T %z')" "$*" >> "$LOG"; }
check() {
  local s=0 model round group n
  n=$(find "$ROOT/formal_runs" -mindepth 2 -maxdepth 2 -type f -name .SUCCESS 2>/dev/null | wc -l)
  [ "$n" -eq 48 ] || { echo "formal_success=$n/48"; s=1; }
  for model in daily gru15 mixed incremental; do
    for round in 2024H2 2025H1 2025H2 2026H1; do
      group="$ROOT/formal_feature_valid/${model}_${round}"
      [ -f "$group/ensemble.npz" ] || { echo "missing_feature_ensemble=${model}_${round}"; s=1; }
      [ -f "$group/POST_SUCCESS" ] || { echo "missing_post=${model}_${round}"; s=1; }
      [ -f "$group/portfolio_0/portfolio_summary.json" ] || { echo "missing_portfolio0=${model}_${round}"; s=1; }
      [ -f "$group/portfolio_10/portfolio_summary.json" ] || { echo "missing_portfolio10=${model}_${round}"; s=1; }
    done
  done
  [ -f "$ROOT/formal_feature_valid/final_report.md" ] || { echo "missing_final_report"; s=1; }
  [ -f "$OUT/FINAL_RESULTS_COMPLETE" ] || { echo "missing_FINAL_RESULTS_COMPLETE"; s=1; }
  [ -f "$OUT/RESULT_BUNDLE.tar.gz" ] || { echo "missing_RESULT_BUNDLE"; s=1; }
  [ -f "$OUT/RESULT_BUNDLE.sha256" ] || { echo "missing_RESULT_BUNDLE.sha256"; s=1; }
  if [ -f "$OUT/RESULT_BUNDLE.tar.gz" ] && [ -f "$OUT/RESULT_BUNDLE.sha256" ]; then
    (cd "$OUT" && sha256sum -c RESULT_BUNDLE.sha256 >/dev/null 2>&1) || { echo "bundle_sha_mismatch"; s=1; }
  fi
  [ -f "$OUT/LOCAL_EXPORT_VERIFIED" ] || { echo "missing_LOCAL_EXPORT_VERIFIED"; s=1; }
  if [ -f "$OUT/LOCAL_EXPORT_VERIFIED" ] && [ -f "$OUT/RESULT_BUNDLE.sha256" ]; then
    grep -Fq "$(awk '{print $1}' "$OUT/RESULT_BUNDLE.sha256")" "$OUT/LOCAL_EXPORT_VERIFIED" || { echo "local_export_hash_mismatch"; s=1; }
  fi
  return "$s"
}

if [ "$MODE" = once ]; then
  if out=$(check 2>&1); then log "DRY_RUN_READY $out"; else log "DRY_RUN_NOT_READY $out"; fi
  cat "$LOG"
  exit 0
fi

log "GUARD_STARTED pid=$$ interval=$INTERVAL root=$ROOT pid1=$(ps -p 1 -o comm= 2>/dev/null || true)"
while :; do
  if out=$(check 2>&1); then
    if [ -f "$OUT/SHUTDOWN_TRIGGERED" ]; then log "ALREADY_TRIGGERED"; exit 0; fi
    log "ALL_CONDITIONS_MET $out"
    log "AUTO_SHUTDOWN_DISABLED_BY_USER; leaving server running"
    exit 0
  fi
  log "NOT_READY $out"
  sleep "$INTERVAL"
done
