#!/usr/bin/env python3
"""Point-in-time stock filters and industry-only factor post-processing.

This is deliberately downstream of prediction generation.  It uses only
metadata whose effective/announcement date is no later than feature_date.
Market-cap neutralization is not attempted because no market-cap field exists
in the supplied data.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def _date(x): return pd.Timestamp(int(x), unit="D") if False else pd.to_datetime(str(int(x)))


def process(pred_path, out_path, data_root, codes_path, mad_z=5.0, industry_src="SW2021"):
    z = np.load(pred_path, allow_pickle=False)
    for k in ("pred", "feature_date", "valid"):
        if k not in z.files: raise ValueError(f"prediction file missing {k}")
    direct_codes = "stock_code" in z.files
    if direct_codes:
        code = np.asarray(z["stock_code"]).astype(str)
        codes = np.unique(code)
        stock = np.zeros(len(code), dtype=int)
    elif "stock" not in z.files:
        raise ValueError("prediction requires stock_code or codes+stock")
    elif "codes" in z.files:
        codes = np.asarray(z["codes"]).astype(str)
    else:
        codes = pd.read_csv(codes_path, header=None).iloc[:, 0].astype(str).to_numpy()
    pred = np.asarray(z["pred"], float).copy(); fdate = np.asarray(z["feature_date"]); valid = np.asarray(z["valid"], bool).copy(); stock=np.asarray(z["stock"], int) if not direct_codes else stock
    if stock.max(initial=-1) >= len(codes): raise ValueError("stock id exceeds explicit codes mapping")
    if not direct_codes: code = codes[stock]
    dates = pd.to_datetime(fdate.astype(str), format="%Y%m%d")
    root=Path(data_root); basic=pd.read_parquet(root/"meta/stock_basic.parquet"); listing=basic.set_index("ts_code")
    names=pd.read_parquet(root/"meta/name.parquet"); names["start_date"]=pd.to_datetime(names.start_date); names["end_date"]=pd.to_datetime(names.end_date)
    names["ann_date"]=pd.to_datetime(names.ann_date); names=names.sort_values(["ts_code","ann_date","start_date"])
    criteria=pd.read_parquet(root/"meta/industry_criteria.parquet"); l1=set(criteria.loc[(criteria.level=="L1") & (criteria.src==industry_src),"index_code"])
    member=pd.read_parquet(root/"meta/sw_index_member.parquet"); member["in_date"]=pd.to_datetime(member.in_date); member["out_date"]=pd.to_datetime(member.out_date)
    name_by_code={c:g for c,g in names.groupby("ts_code",sort=False)}
    member_by_code={c:g for c,g in member[member.index_code.isin(l1)].groupby("ts_code",sort=False)}
    keep=np.ones(len(pred),bool); industry=np.full(len(pred),None,dtype=object)
    for i,(c,dt) in enumerate(zip(code,dates)):
        if c not in listing.index: keep[i]=False; continue
        r=listing.loc[c]
        if pd.notna(r.list_date) and pd.Timestamp(r.list_date)>dt: keep[i]=False
        if pd.notna(r.delist_date) and pd.Timestamp(r.delist_date)<dt: keep[i]=False
        ng=name_by_code.get(c)
        nr=ng[(ng.start_date<=dt)&(ng.ann_date<=dt)&(ng.end_date.isna()| (ng.end_date>=dt))] if ng is not None else ng
        if nr is not None and len(nr):
            nm=str(nr.iloc[-1]["name"])
            if "ST" in nm.upper() or "PT" in nm.upper() or "退" in nm: keep[i]=False
        mg=member_by_code.get(c)
        mr=mg[(mg.in_date<=dt)&(mg.out_date.isna()|(mg.out_date>=dt))] if mg is not None else mg
        if mr is not None and len(mr): industry[i]=sorted(mr.index_code.astype(str))[0]
    valid &= keep
    # Industry-only output excludes stocks with no point-in-time L1 mapping;
    # they remain available through pred_raw for separate diagnostics.
    valid &= pd.notna(industry)
    raw=pred.copy(); neutral=pred.copy(); wins=0; winsorized_stocks=0; no_ind=0
    for dt in pd.unique(dates):
        ix=np.flatnonzero(valid & (dates==dt)); groups={}
        for i in ix:
            groups.setdefault(industry[i],[]).append(i)
        if len(ix) == 0: continue
        xall=pred[ix]; med=np.nanmedian(xall); mad=np.nanmedian(np.abs(xall-med)); scale=1.4826*mad
        clipped_all=np.clip(xall,med-mad_z*scale,med+mad_z*scale) if np.isfinite(scale) and scale>0 else xall
        winsorized_stocks += int(np.sum(clipped_all != xall))
        clipped_by_i={i:v for i,v in zip(ix,clipped_all)}
        residuals={}
        for g in groups.values():
            clipped=np.array([clipped_by_i[i] for i in g])
            residuals.update({i: v-np.nanmean(clipped) for i,v in zip(g,clipped)})
        rv=np.array(list(residuals.values())); rs=np.nanstd(rv); rm=np.nanmean(rv)
        for i,v in residuals.items(): neutral[i]=(v-rm)/rs if np.isfinite(rs) and rs>0 else v-rm
    no_ind=int(np.sum(valid==False)-np.sum((~keep)&np.asarray(z["valid"],bool)))
    out=Path(out_path); out.parent.mkdir(parents=True,exist_ok=True)
    fields=dict(pred=neutral.astype(np.float32), pred_raw=raw.astype(np.float32), feature_date=fdate, valid=valid)
    fields["stock_code"] = code.astype("U")
    fields["stock"] = stock; fields["codes"] = np.asarray(codes, dtype="U")
    np.savez_compressed(out, **fields)
    stats={"rows":int(len(pred)),"input_valid":int(np.asarray(z["valid"],bool).sum()),"output_valid":int(valid.sum()),"filtered_stock_status":int((np.asarray(z["valid"],bool)&~keep).sum()),"no_l1_industry":int(np.sum(keep & ~pd.notna(industry))),"industry_src":industry_src,"industry_level":industry_src+" L1","winsorized_stocks":winsorized_stocks,"winsorization":"Each date's full valid cross-section, MAD cutoff 5.0 scaled MAD (implementation assumption; report says winsorization but gives no rule)","neutralization":"industry-group mean residual plus group standardization; stocks without L1 mapping excluded from industry-only valid and retained only in pred_raw; market-cap neutralization unavailable"}
    out.with_suffix(".json").write_text(json.dumps(stats,indent=2,default=str),encoding="utf-8"); return stats

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--pred",required=True); p.add_argument("--out",required=True); p.add_argument("--data-root",required=True); p.add_argument("--codes",required=True); p.add_argument("--industry-src",default="SW2021"); a=p.parse_args(); print(json.dumps(process(a.pred,a.out,a.data_root,a.codes,industry_src=a.industry_src),indent=2))
