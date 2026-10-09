#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/formal_runs" "$T/formal_feature_valid" "$T/finalization"
for i in $(seq -w 1 48); do mkdir -p "$T/formal_runs/run_${i}"; touch "$T/formal_runs/run_${i}/.SUCCESS"; done
for model in daily gru15 mixed incremental; do
  for round in 2024H2 2025H1 2025H2 2026H1; do
    d="$T/formal_feature_valid/${model}_${round}"
    mkdir -p "$d/portfolio_0" "$d/portfolio_10"
    touch "$d/ensemble.npz" "$d/POST_SUCCESS" "$d/portfolio_0/portfolio_summary.json" "$d/portfolio_10/portfolio_summary.json"
  done
done
touch "$T/formal_feature_valid/final_report.md" "$T/finalization/FINAL_RESULTS_COMPLETE"
printf bundle > "$T/finalization/RESULT_BUNDLE.tar.gz"
(cd "$T/finalization" && sha256sum RESULT_BUNDLE.tar.gz > RESULT_BUNDLE.sha256)

calls="$T/calls"
cat > "$T/mock_shutdown" <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "${MOCK_CALLS:?}"
EOF
chmod +x "$T/mock_shutdown"

# Missing local export must block the diagnostic completion gate.
AUTO_SHUTDOWN=1 SHUTDOWN_BIN="$T/mock_shutdown" MOCK_CALLS="$calls" "$ROOT/auto_shutdown_guard.sh" "$T" once >/dev/null
[ ! -e "$calls" ] || { echo "FAIL: missing export gate invoked mock" >&2; exit 1; }

sha=$(awk '{print $1}' "$T/finalization/RESULT_BUNDLE.sha256")
echo "bundle_sha256=$sha" > "$T/finalization/LOCAL_EXPORT_VERIFIED"
AUTO_SHUTDOWN=1 SHUTDOWN_BIN="$T/mock_shutdown" MOCK_CALLS="$calls" "$ROOT/auto_shutdown_guard.sh" "$T" watch >/dev/null
[ ! -e "$calls" ] || { echo "FAIL: revoked shutdown path invoked mock" >&2; exit 1; }

# A second watcher remains inert.
AUTO_SHUTDOWN=1 SHUTDOWN_BIN="$T/mock_shutdown" MOCK_CALLS="$calls" "$ROOT/auto_shutdown_guard.sh" "$T" watch >/dev/null
[ ! -e "$calls" ] || { echo "FAIL: second disabled watcher invoked mock" >&2; exit 1; }
echo "PASS: missing gate blocks; revoked shutdown path stays inert"
