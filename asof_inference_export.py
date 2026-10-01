"""Export an as-of prediction pool without using future labels.

The existing ensemble NPZ already contains checkpoint predictions.  This
utility only restricts by feature_date and preserves prediction values; y and
label_end are retained solely for optional diagnostics, never for valid mask.
"""
import argparse, json
from pathlib import Path
import numpy as np

def export_asof(src, dst, start=None, end=None):
    z=np.load(src, allow_pickle=False)
    need={"pred","feature_date","valid"}
    if not need.issubset(z.files): raise ValueError(f"missing {need-set(z.files)}")
    fd=np.asarray(z["feature_date"]); mask=np.asarray(z["valid"],bool).copy()
    if start is not None: mask &= fd >= int(start)
    if end is not None: mask &= fd <= int(end)
    # Keep row alignment and values; downstream code consumes valid as the
    # feature-available inference pool. Label validity is diagnostic only.
    fields={k:z[k] for k in z.files}; fields["valid_asof"]=mask; fields["valid"]=mask
    if "label_end" in z.files:
        fields["label_valid_diagnostic"]=np.asarray(z["label_end"]) <= (int(end) if end else 99991231)
    np.savez_compressed(dst,**fields)
    report={"rows":len(fd),"feature_valid_rows":int(mask.sum()),"label_used_for_valid":False,"pred_values_preserved":True,"start":start,"end":end}
    Path(dst).with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--src',required=True); p.add_argument('--out',required=True); p.add_argument('--start',type=int); p.add_argument('--end',type=int); a=p.parse_args(); print(json.dumps(export_asof(a.src,a.out,a.start,a.end),indent=2))
