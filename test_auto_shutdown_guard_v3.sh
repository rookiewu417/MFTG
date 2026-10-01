#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/formal_runs" "$T/formal_feature_valid" "$T/finalization"
for r in 2024H2 2025H1 2025H2 2026H1; do for m in daily gru15 mixed incremental; do for s in 7 11 42; do
  d="$T/formal_runs/${m}_${r}_seed${s}"; mkdir -p "$d"
  touch "$d/.SUCCESS" "$d/${m}_${r}.pt" "$d/${m}_${r}_pred.npz"
  printf 'model=%s round=%s seed=%s epochs=100 patience=10 batch=5000 every=5 data_interval=x cache_sha=x\n' "$m" "$r" "$s" > "$d/config.txt"
  [ "$m" = incremental ] && echo 'logic=incremental-stage1-v2' >> "$d/config.txt"
done; done; done
for m in daily gru15 mixed incremental; do for r in 2024H2 2025H1 2025H2 2026H1; do
  d="$T/formal_feature_valid/${m}_${r}"; mkdir -p "$d/portfolio_0" "$d/portfolio_10"
  touch "$d/ensemble.npz" "$d/POST_SUCCESS" "$d/portfolio_0/portfolio_summary.json" "$d/portfolio_10/portfolio_summary.json"
done; done
touch "$T/formal_feature_valid/final_report.md"
RID='test_unique_run_123'; printf 'run_id=%s\n' "$RID" > "$T/finalization/GUARD_RUN_ID"
printf 'run_id=%s\nmanifest_sha256=placeholder\n' "$RID" > "$T/finalization/FINAL_RESULTS_COMPLETE"
printf bundle > "$T/finalization/RESULT_BUNDLE.tar.gz"
(cd "$T/finalization" && sha256sum RESULT_BUNDLE.tar.gz > RESULT_BUNDLE.sha256 && sha256sum finalization_manifest.json 2>/dev/null || true)
printf '{}' > "$T/finalization/finalization_manifest.json"
msha=$(sha256sum "$T/finalization/finalization_manifest.json" | awk '{print $1}')
bsha=$(awk '{print $1}' "$T/finalization/RESULT_BUNDLE.sha256")
printf 'run_id=%s\nmanifest_sha256=%s\n' "$RID" "$msha" > "$T/finalization/FINAL_RESULTS_COMPLETE"
printf 'run_id=%s\nbundle_sha256=%s\nmanifest_sha256=%s\n' "$RID" "$bsha" "$msha" > "$T/finalization/LOCAL_EXPORT_VERIFIED"
printf 'run_id=%s\n' "$RID" > "$T/finalization/ARM_$RID"
printf 'run_id=old\n' > "$T/finalization/ARM_old"
cat > "$T/mock_shutdown" <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "${MOCK_CALLS:?}"
EOF
chmod +x "$T/mock_shutdown"
# Wrong run id and stale ARM must remain blocked.
RUN_ID=other SHUTDOWN_BIN="$T/mock_shutdown" "$ROOT/auto_shutdown_guard_v3.sh" "$T" once >/dev/null
[ ! -e "$T/calls" ] || { echo 'FAIL: stale ARM accepted'; exit 1; }
# Correct complete gate invokes only the injected mock once.
RUN_ID="$RID" SHUTDOWN_BIN="$T/mock_shutdown" MOCK_CALLS="$T/calls" "$ROOT/auto_shutdown_guard_v3.sh" "$T" watch >/dev/null
[ "$(wc -l < "$T/calls")" -eq 1 ] || { echo 'FAIL: complete gate did not invoke once'; exit 1; }
RUN_ID="$RID" SHUTDOWN_BIN="$T/mock_shutdown" MOCK_CALLS="$T/calls" "$ROOT/auto_shutdown_guard_v3.sh" "$T" watch >/dev/null
[ "$(wc -l < "$T/calls")" -eq 1 ] || { echo 'FAIL: duplicate invocation'; exit 1; }
echo 'PASS: exact run/SHA/manifest gates and one-shot mock shutdown'
