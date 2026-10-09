#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
MODE="${2:-watch}"
RUN_ID="${RUN_ID:-}"
FINAL="$ROOT/finalization"
INTERVAL="${GUARD_INTERVAL:-60}"
SHUTDOWN_BIN="${SHUTDOWN_BIN:-/usr/bin/shutdown}"
if [ -z "$RUN_ID" ]; then echo "RUN_ID must be explicitly supplied" >&2; exit 2; fi
LOG="$FINAL/auto_shutdown_guard_${RUN_ID}.log"
mkdir -p "$FINAL"
log(){ printf '%s %s\n' "$(date '+%F %T %z')" "$*" >> "$LOG"; }
expected_runs_ok(){
  local m r s d n=0 prefix
  for r in 2024H2 2025H1 2025H2 2026H1; do for m in daily gru15 mixed incremental; do for s in 7 11 42; do
    d="$ROOT/formal_runs/${m}_${r}_seed${s}"
    [ -s "$d/.SUCCESS" ] && [ -s "$d/${m}_${r}.pt" ] && [ -s "$d/${m}_${r}_pred.npz" ] || { echo "bad_run=${m}_${r}_seed${s}"; return 1; }
    prefix="model=${m} round=${r} seed=${s} epochs=100 patience=10 batch=5000 every=5"
    grep -Fq "$prefix" "$d/config.txt" || { echo "bad_config=${m}_${r}_seed${s}"; return 1; }
    [ "$m" != incremental ] || grep -Fq "logic=incremental-stage1-v2" "$d/config.txt" || { echo "bad_logic=${m}_${r}_seed${s}"; return 1; }
    n=$((n+1))
  done; done; done
  [ "$n" -eq 48 ] || { echo "run_count=$n"; return 1; }
}
feature_outputs_ok(){
  local m r d
  for m in daily gru15 mixed incremental; do for r in 2024H2 2025H1 2025H2 2026H1; do
    d="$ROOT/formal_feature_valid/${m}_${r}"
    [ -s "$d/ensemble.npz" ] && [ -s "$d/POST_SUCCESS" ] || { echo "missing_feature_group=${m}_${r}"; return 1; }
    [ -s "$d/portfolio_0/portfolio_summary.json" ] && [ -s "$d/portfolio_10/portfolio_summary.json" ] || { echo "missing_portfolio=${m}_${r}"; return 1; }
  done; done
}
check(){
  expected_runs_ok || return 1
  feature_outputs_ok || return 1
  [ -s "$ROOT/formal_feature_valid/final_report.md" ] || { echo "missing_report"; return 1; }
  [ -s "$FINAL/finalization_manifest.json" ] || { echo "missing_manifest"; return 1; }
  [ -s "$FINAL/FINAL_RESULTS_COMPLETE" ] || { echo "missing_final_marker"; return 1; }
  grep -Fxq "run_id=$RUN_ID" "$FINAL/FINAL_RESULTS_COMPLETE" || { echo "marker_run_id_mismatch"; return 1; }
  local msha="$(sha256sum "$FINAL/finalization_manifest.json" | awk '{print $1}')"
  grep -Fxq "manifest_sha256=$msha" "$FINAL/FINAL_RESULTS_COMPLETE" || { echo "marker_manifest_mismatch"; return 1; }
  [ -s "$FINAL/RESULT_BUNDLE.tar.gz" ] && [ -s "$FINAL/RESULT_BUNDLE.sha256" ] || { echo "missing_bundle"; return 1; }
  (cd "$FINAL" && sha256sum -c RESULT_BUNDLE.sha256 >/dev/null 2>&1) || { echo "bundle_sha_mismatch"; return 1; }
  local bsha="$(awk '{print $1}' "$FINAL/RESULT_BUNDLE.sha256")"
  [ -s "$FINAL/LOCAL_EXPORT_VERIFIED" ] || { echo "missing_local_export"; return 1; }
  grep -Fxq "run_id=$RUN_ID" "$FINAL/LOCAL_EXPORT_VERIFIED" || { echo "export_run_id_mismatch"; return 1; }
  grep -Fxq "bundle_sha256=$bsha" "$FINAL/LOCAL_EXPORT_VERIFIED" || { echo "export_sha_mismatch"; return 1; }
  grep -Fxq "manifest_sha256=$msha" "$FINAL/LOCAL_EXPORT_VERIFIED" || { echo "export_manifest_mismatch"; return 1; }
  [ -s "$FINAL/ARM_$RUN_ID" ] || { echo "not_armed"; return 1; }
  grep -Fxq "run_id=$RUN_ID" "$FINAL/ARM_$RUN_ID" || { echo "arm_token_mismatch"; return 1; }
}
if [ "$MODE" = once ]; then
  if out="$(check 2>&1)"; then log "DRYRUN_READY run_id=$RUN_ID $out"; else log "DRYRUN_BLOCKED run_id=$RUN_ID $out"; fi
  cat "$LOG"; exit 0
fi
log "WATCH_STARTED pid=$$ run_id=$RUN_ID armed=$([ -f "$FINAL/ARM_$RUN_ID" ] && echo yes || echo no)"
while :; do
  if [ ! -f "$FINAL/ARM_$RUN_ID" ]; then log "DISARMED"; sleep "$INTERVAL"; continue; fi
  if out="$(check 2>&1)"; then
    [ ! -e "$FINAL/TRIGGERED_$RUN_ID" ] || { log "ALREADY_TRIGGERED"; exit 0; }
    (set -C; : > "$FINAL/TRIGGERED_$RUN_ID") 2>/dev/null || { log "TRIGGER_RACE_LOST"; exit 0; }
    sync
    log "ALL_GATES_TRUE_EXECUTING_SHUTDOWN run_id=$RUN_ID"
    "$SHUTDOWN_BIN" -h now >> "$LOG" 2>&1
    exit $?
  else log "BLOCKED $out"; fi
  sleep "$INTERVAL"
done
