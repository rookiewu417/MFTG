#!/usr/bin/env bash
set -euo pipefail

# Finalize a completed remote baseline run without including raw data/cache.
ROOT="$(cd "${1:-.}" && pwd)"
OUT="$ROOT/finalization"
mkdir -p "$OUT"
die() { echo "finalize_results: $*" >&2; exit 1; }
need_file() { [ -f "$1" ] || die "missing $1"; }

success_count=$(find "$ROOT/formal_runs" -mindepth 2 -maxdepth 2 -type f -name .SUCCESS | wc -l)
[ "$success_count" -eq 48 ] || die "formal SUCCESS count is $success_count, expected 48"
models=(daily gru15 mixed incremental)
rounds=(2024H2 2025H1 2025H2 2026H1)
for model in "${models[@]}"; do
  for round in "${rounds[@]}"; do
    group="$ROOT/formal_feature_valid/${model}_${round}"
    need_file "$group/ensemble.npz"
    need_file "$group/POST_SUCCESS"
    need_file "$group/portfolio_0/portfolio_summary.json"
    need_file "$group/portfolio_10/portfolio_summary.json"
  done
done
need_file "$ROOT/formal_feature_valid/final_report.md"

manifest_tmp="$OUT/finalization_manifest.json.tmp"
python - "$ROOT" "$manifest_tmp" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
success = sorted(str(p.relative_to(root)) for p in (root / "formal_runs").glob("*_seed*/.SUCCESS"))
groups = []
for p in sorted((root / "formal_feature_valid").glob("*/ensemble.npz")):
    groups.append({"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size})
obj = {"schema": 1, "formal_success": len(success), "success_markers": success,
       "feature_groups": groups, "report": "formal_feature_valid/final_report.md"}
out.write_text(json.dumps(obj, indent=2) + "\n")
PY
mv "$manifest_tmp" "$OUT/finalization_manifest.json"

# Keep code, logs, checkpoints and results; exclude the multi-GB base/cache/envs.
rm -f "$OUT/RESULT_BUNDLE.tar.gz" "$OUT/RESULT_BUNDLE.sha256" "$OUT/FINAL_RESULTS_COMPLETE"
tar -czf "$OUT/RESULT_BUNDLE.tar.gz" \
  --exclude='./full_base' --exclude='./normalized_cache' --exclude='./gpuenv' \
  --exclude='./.pixi' --exclude='./finalization/RESULT_BUNDLE.tar.gz' \
  -C "$ROOT" formal_runs formal_feature_valid formal_ensembles \
  pixi.toml pixi.lock README.md METHOD_LIMITATIONS.md \
  rolling_full.py fullbase_dataset.py gru_baseline.py ensemble_predictions.py \
  feature_valid_inference.py factor_postprocess.py portfolio_backtest.py \
  compare_feature_valid.py 2>/dev/null || die "bundle creation failed"
(cd "$OUT" && sha256sum RESULT_BUNDLE.tar.gz > RESULT_BUNDLE.sha256)
cat > "$OUT/FINAL_RESULTS_COMPLETE" <<EOF
schema=1
formal_success=48
feature_groups=16
run_id=${RUN_ID:?RUN_ID must be set for finalization}
manifest_sha256=$(sha256sum "$OUT/finalization_manifest.json" | awk '{print $1}')
bundle_sha256=$(awk '{print $1}' "$OUT/RESULT_BUNDLE.sha256")
created_at=$(date -Iseconds)
EOF
cp "$OUT/FINAL_RESULTS_COMPLETE" "$OUT/SERVER_EXPORT_READY"
echo "Finalization complete. Export RESULT_BUNDLE.tar.gz and then create LOCAL_EXPORT_VERIFIED containing run_id, bundle_sha256, and manifest_sha256."
