#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
RUN_ID="fixture_run_$(date +%s)"
mkdir -p "$T/formal_runs" "$T/formal_feature_valid" "$T/finalization"
for r in 2024H2 2025H1 2025H2 2026H1; do
  for m in daily gru15 mixed incremental; do
    for s in 7 11 42; do
      d="$T/formal_runs/${m}_${r}_seed${s}"
      mkdir -p "$d"
      printf ok > "$d/.SUCCESS"
      printf ckpt > "$d/${m}_${r}.pt"
      printf pred > "$d/${m}_${r}_pred.npz"
      printf 'epochs=100\n' > "$d/config.txt"
    done
    d="$T/formal_feature_valid/${m}_${r}"
    mkdir -p "$d/portfolio_0" "$d/portfolio_10"
    printf ensemble > "$d/ensemble.npz"
    printf post > "$d/POST_SUCCESS"
    printf '{}' > "$d/portfolio_0/portfolio_summary.json"
    printf '{}' > "$d/portfolio_10/portfolio_summary.json"
  done
done
printf report > "$T/formal_feature_valid/final_report.md"
printf manifest > "$T/finalization/finalization_manifest.json"
printf bundle > "$T/finalization/RESULT_BUNDLE.tar.gz"
(cd "$T/finalization" && sha256sum RESULT_BUNDLE.tar.gz > RESULT_BUNDLE.sha256)
manifest_sha="$(sha256sum "$T/finalization/finalization_manifest.json" | awk '{print $1}')"
bundle_sha="$(awk '{print $1}' "$T/finalization/RESULT_BUNDLE.sha256")"
printf 'run_id=%s\nmanifest_sha256=%s\n' "$RUN_ID" "$manifest_sha" > "$T/finalization/FINAL_RESULTS_COMPLETE"
printf 'run_id=%s\nbundle_sha256=%s\nmanifest_sha256=%s\n' "$RUN_ID" "$bundle_sha" "$manifest_sha" > "$T/finalization/LOCAL_EXPORT_VERIFIED"

# Dry-run is never allowed to invoke a shutdown binary.
RUN_ID="$RUN_ID" SHUTDOWN_BIN="$T/should_not_run" bash "$ROOT_DIR/auto_shutdown_guard_v2.sh" "$T" once >/dev/null

# Wrong/stale token must block before the mock is reached.
printf 'run_id=%s\nbundle_sha256=stale\nmanifest_sha256=%s\n' "$RUN_ID" "$manifest_sha" > "$T/finalization/ARM_$RUN_ID"
printf '#!/usr/bin/env bash\nprintf called > "%s"\n' "$T/called" > "$T/mock_shutdown"
chmod +x "$T/mock_shutdown"
if RUN_ID="$RUN_ID" GUARD_INTERVAL=0 SHUTDOWN_BIN="$T/mock_shutdown" bash "$ROOT_DIR/auto_shutdown_guard_v2.sh" "$T" watch; then :; fi
[ ! -e "$T/called" ] || { echo 'FAIL stale ARM token invoked mock'; exit 1; }

# Correct token invokes the injected mock once and creates a trigger marker.
printf 'run_id=%s\nbundle_sha256=%s\nmanifest_sha256=%s\n' "$RUN_ID" "$bundle_sha" "$manifest_sha" > "$T/finalization/ARM_$RUN_ID"
RUN_ID="$RUN_ID" GUARD_INTERVAL=0 SHUTDOWN_BIN="$T/mock_shutdown" bash "$ROOT_DIR/auto_shutdown_guard_v2.sh" "$T" watch >/dev/null
[ -e "$T/called" ] || { echo 'FAIL valid ARM token did not invoke mock'; cat "$T/finalization/auto_shutdown_guard_${RUN_ID}.log"; exit 1; }
[ -e "$T/finalization/TRIGGERED_$RUN_ID" ] || { echo 'FAIL trigger marker missing'; exit 1; }
rm -f "$T/called"
RUN_ID="$RUN_ID" GUARD_INTERVAL=0 SHUTDOWN_BIN="$T/mock_shutdown" bash "$ROOT_DIR/auto_shutdown_guard_v2.sh" "$T" watch >/dev/null
[ ! -e "$T/called" ] || { echo 'FAIL second watch invoked mock'; exit 1; }
echo 'PASS v2 dry-run, stale-token block, valid mock trigger, one-shot trigger'
