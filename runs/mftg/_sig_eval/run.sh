#!/usr/bin/env bash
# Signal-level re-evaluation (2026-10-31 .. 11-01): the exact runs behind runs/mftg/_sig_eval/*.
# Run on the remote machine from the mirror ~/remote/projects/MFTG (runs/ there holds the pred files; build/ holds shared base + styles/extra).
# Usage: bash runs/mftg/_sig_eval/run.sh [main|prune|pt_mkt|styles|layers ...]   (default: all); outputs land in ~/remote/<name>
set -euo pipefail
cd ~/remote/projects/MFTG
export MFTG_OUT=$PWD/build OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$HOME/mftg_rebal/.venv/bin/python -W ignore"
D=Daily-Neu H=Daily-Neu-H256 C=Daily-Neu-CSI800 F=Full-Neu M=_merge E=e2e

run() {
  case $1 in
  main)   # all candidates, three independent seed sets, single members   (sig_eval: IC, neutral IC, deciles, decay, pools, increment)
    $PY mftg/sig_eval.py --out ~/remote/sig_eval --jobs 28 \
      NEU4=$D+$H+$C+$F NEU4_6=$D+$M/H256_6+$M/C800_6+$F DMV6=$M/dDMV6+$H+$C+$F R12=$M/dDMV6+$H+$C+$M/fVOL6 CAND=$M/dDMV9+$M/H256_6+$M/C800_6+$M/fVOL9 \
      CAND_s1=$E/dDMV+$H+$C+$E/fVOLf CAND_s2=$E/dDMV_s2+$E/H256_s2+$E/C800_s2+$E/fVOLf_s2 CAND_s3=$E/dDMV_s3+$H+$C+$E/fVOLf_s3 \
      NEU4+dX=$D+$H+$C+$F+$E/dX dX_D=$E/dX+$H+$C+$F fDMV_F=$D+$H+$C+$E/fDMVf CAND+dX=$M/dDMV9+$M/H256_6+$M/C800_6+$M/fVOL9+$E/dX \
      m_D=$D m_H=$H m_C=$C m_F=$F m_dDMV9=$M/dDMV9 m_fVOL9=$M/fVOL9 m_H6=$M/H256_6 m_C6=$M/C800_6 m_dX=$E/dX \
      m_dDMV_a=$E/dDMV m_dDMV_b=$E/dDMV_s2 m_dDMV_c=$E/dDMV_s3 m_fVOL_a=$E/fVOLf m_fVOL_b=$E/fVOLf_s2 m_fVOL_c=$E/fVOLf_s3 ;;
  prune)  # dropping members of the first candidate
    $PY mftg/sig_eval.py --out ~/remote/sig_eval2 --jobs 12 \
      NEU4=$D+$H+$C+$F CAND=$M/dDMV9+$M/H256_6+$M/C800_6+$M/fVOL9 CAND-C=$M/dDMV9+$M/H256_6+$M/fVOL9 CAND-H=$M/dDMV9+$M/C800_6+$M/fVOL9 \
      CAND-HC=$M/dDMV9+$M/fVOL9 CAND+F=$M/dDMV9+$M/H256_6+$M/C800_6+$M/fVOL9+$F CAND-C+dX=$M/dDMV9+$M/H256_6+$M/fVOL9+$E/dX ;;
  pt_mkt) # PatchTST and market-gate members
    $PY mftg/sig_eval.py --out ~/remote/sig_eval3 --jobs 6 \
      NEU4=$D+$H+$C+$F PT_D=$E/dPT+$H+$C+$F MK2_H=$D+$E/dMK2H256+$C+$F m_dPT=$E/dPT m_dMK2H=$E/dMK2H256 ;;
  styles) # 14-style exposure, IC decomposition, pure factor return, industry tilt
    $PY mftg/sty_eval.py --out ~/remote/sty_eval --jobs 9 \
      NEU4=$D+$H+$C+$F CAND=$M/dDMV9+$M/H256_6+$M/C800_6+$M/fVOL9 CAND-C=$M/dDMV9+$M/H256_6+$M/fVOL9 CAND-C+dX=$M/dDMV9+$M/H256_6+$M/fVOL9+$E/dX \
      m_D=$D m_F=$F m_dDMV9=$M/dDMV9 m_fVOL9=$M/fVOL9 m_dX=$E/dX ;;
  layers) # additive per-style IC shares, raw / pure deciles, IC inside style layers
    $PY mftg/sty_layer.py --out ~/remote/sty_layer --jobs 4 \
      NEU4=$D+$H+$C+$F CAND=$M/dDMV9+$M/H256_6+$M/C800_6+$M/fVOL9 CAND-C=$M/dDMV9+$M/H256_6+$M/fVOL9 CAND-C+dX=$M/dDMV9+$M/H256_6+$M/fVOL9+$E/dX ;;
  *) echo "unknown run $1" >&2; exit 1 ;;
  esac
}

for r in "${@:-main prune pt_mkt styles layers}"; do for x in $r; do run "$x" > ~/remote/logs/sig_$x.log 2>&1; done; done
