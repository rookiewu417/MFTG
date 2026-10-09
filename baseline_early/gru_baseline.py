#!/usr/bin/env python3
"""Multi-frequency GRU baseline."""
from __future__ import annotations
import argparse, glob, json, os, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import polars as pl
from scipy.stats import spearmanr
import torch
from torch import nn

DEFAULT_DATA="/root/autodl-tmp/quant/data/ashare"
DEFAULT_OUT="/root/autodl-tmp/quant/repro_ml_factor_luna_20260917/artifacts"
FEATS=["open","high","low","close","vwap","vol"]

def parse():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd",required=True)
    m=sub.add_parser("metadata"); m.add_argument("--data-root",default=DEFAULT_DATA)
    q=sub.add_parser("prepare"); q.add_argument("--data-root",default=DEFAULT_DATA); q.add_argument("--out",default=DEFAULT_OUT)
    q.add_argument("--start",default="2016-01-04"); q.add_argument("--end",default="2026-06-30")
    q.add_argument("--max-stocks",type=int,default=512); q.add_argument("--every",type=int,default=5)
    t=sub.add_parser("train"); t.add_argument("--out",default=DEFAULT_OUT); t.add_argument("--model",choices=["daily","gru15","mixed","incremental"],default="gru15")
    t.add_argument("--epochs",type=int,default=20); t.add_argument("--patience",type=int,default=5); t.add_argument("--seed",type=int,default=7)
    e=sub.add_parser("evaluate"); e.add_argument("--out",default=DEFAULT_OUT); e.add_argument("--model",default="gru15")
    return p.parse_args()

def metadata(root):
    root=Path(root); minute=sorted(root.glob("minute/bars_1min/year=*/month=*/data.parquet"))
    print(json.dumps({"data_root":str(root),"daily":str(root/"daily/post_weighted.parquet"),
      "daily_rows":pq.ParquetFile(root/"daily/post_weighted.parquet").metadata.num_rows,
      "minute_partitions":len(minute),"first_minute":str(minute[0]) if minute else None,
      "last_minute":str(minute[-1]) if minute else None,
      "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},ensure_ascii=False,indent=2))

def zscore_cs(x):
    mu=np.nanmean(x,axis=0,keepdims=True); sd=np.nanstd(x,axis=0,keepdims=True)
    return np.nan_to_num((x-mu)/np.where(sd<1e-6,1,sd),nan=0,posinf=0,neginf=0).astype(np.float32)

def prep_seq(x, denom_last=False):
    if denom_last: den=x[:,-1:,:]
    else: den=np.nanmean(x,axis=1,keepdims=True)
    den=np.where(np.abs(den)<1e-8,1,den)
    return zscore_cs(np.nan_to_num(x/den,nan=0,posinf=0,neginf=0))

def aggregate_month(path,codes):
    t0=time.time()
    lf=pl.read_parquet(path,columns=["ts_code","trade_time","open","high","low","close","vol","amount"])
    lf=lf.filter(pl.col("ts_code").is_in(list(codes)))
    lf=lf.with_columns([pl.col("trade_time").dt.date().alias("trade_date"),
      (pl.col("trade_time").dt.hour().cast(pl.Int32)*60+pl.col("trade_time").dt.minute().cast(pl.Int32)).alias("minute")])
    lf=lf.with_columns(pl.when((pl.col("minute")>=571)&(pl.col("minute")<=690)).then((pl.col("minute")-571)//15)
       .when((pl.col("minute")>=781)&(pl.col("minute")<=900)).then(8+(pl.col("minute")-781)//15)
       .otherwise(-1).alias("slot")).filter(pl.col("slot")>=0)
    out=(lf.sort(["ts_code","trade_time"]).group_by(["ts_code","trade_date","slot"],maintain_order=False)
      .agg([pl.col("open").first().alias("open"),pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),pl.col("close").last().alias("close"),
        pl.col("vol").sum().alias("vol"),pl.col("amount").sum().alias("amount")])
      .with_columns((pl.col("amount")/pl.col("vol").replace(0,None)).alias("vwap"))
      .select(["ts_code","trade_date","slot"]+FEATS).sort(["trade_date","ts_code","slot"]))
    print("aggregated",Path(path).parent.parent.name,Path(path).parent.name,"rows",out.height,"sec",round(time.time()-t0,1),flush=True)
    return out.to_pandas()

def prepare(a):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    root=Path(a.data_root); start=pd.Timestamp(a.start); end=pd.Timestamp(a.end)
    d=pd.read_parquet(root/"daily/post_weighted.parquet",columns=["trade_date","ts_code"]+FEATS+["adj_factor"])
    d["trade_date"]=pd.to_datetime(d.trade_date); d=d[(d.trade_date>=start)&(d.trade_date<=end)]
    dates=pd.DatetimeIndex(sorted(d.trade_date.unique()))
    codes=np.array(sorted(d.ts_code.unique())[:a.max_stocks])
    d=d[d.ts_code.isin(codes)].sort_values(["trade_date","ts_code"])
    print("daily rows",len(d),"dates",len(dates),"stocks",len(codes),flush=True)
    def panel(col):
        return d.pivot(index="trade_date",columns="ts_code",values=col).reindex(index=dates,columns=codes).to_numpy(np.float32)
    daily=np.stack([panel(x) for x in FEATS],axis=-1); adj=panel("adj_factor")
    bars=np.full((len(dates),len(codes),16,6),np.nan,np.float32)
    cidx={c:i for i,c in enumerate(codes)}; didx={x:i for i,x in enumerate(dates)}
    mroot=root/"minute/bars_1min"; paths=sorted(mroot.glob("year=*/month=*/data.parquet"))
    for p in paths:
        ym=re.search(r"year=(\d{4})/month=(\d{2})",str(p)); pm=pd.Timestamp(ym.group(1)+"-"+ym.group(2)+"-01")
        if not (pm + pd.offsets.MonthEnd(1) >= start and pm <= end): continue
        x=aggregate_month(str(p),codes); x["trade_date"]=pd.to_datetime(x.trade_date); x=x[x.trade_date.isin(didx)]
        if len(x):
            ii=x.trade_date.map(didx).to_numpy(); jj=x.ts_code.map(cidx).to_numpy(); kk=x.slot.to_numpy()
            vals=x[FEATS].to_numpy(np.float32); vals[:,:5]*=np.nan_to_num(adj[ii,jj,None],nan=1.0)
            bars[ii,jj,kk,:]=vals
    first=max(40,20); last=len(dates)-10; sidx=np.arange(first,last,a.every,dtype=np.int32)
    N=len(sidx)*len(codes); x15=np.empty((N,320,6),np.float32); xd=np.empty((N,40,6),np.float32)
    yz=np.empty(N,np.float32); ret=np.empty(N,np.float32); validmask=np.empty(N,bool); fdate=np.empty(N,np.int32); lend=np.empty(N,np.int32); stock=np.empty(N,np.int32)
    k=0
    for i in sidx:
        m=prep_seq(bars[i-19:i+1].transpose(1,0,2,3).reshape(len(codes),320,6),False)
        q=prep_seq(daily[i-39:i+1].transpose(1,0,2).reshape(len(codes),40,6),True)
        r=daily[i+10,:,3]/daily[i,:,3]-1; valid=np.isfinite(r) & np.isfinite(bars[i-19:i+1]).all(axis=(0,2,3)) & np.isfinite(daily[i-39:i+1]).all(axis=(0,2)); rz=np.full(len(codes),np.nan,np.float32)
        if valid.any(): rz[valid]=((r[valid]-np.nanmean(r[valid]))/(np.nanstd(r[valid])+1e-8)).astype(np.float32)
        sl=slice(k,k+len(codes)); x15[sl]=m; xd[sl]=q; yz[sl]=np.nan_to_num(rz); ret[sl]=np.nan_to_num(r)
        fdate[sl]=int(dates[i].strftime("%Y%m%d")); lend[sl]=int(dates[i+10].strftime("%Y%m%d")); stock[sl]=np.arange(len(codes)); validmask[sl]=valid; k+=len(codes)
    np.savez(out/"prepared.npz",x15=x15,xd=xd,y=yz,ret=ret,valid=validmask,feature_date=fdate,label_end=lend,stock=stock)
    info={"data_start":str(dates[0].date()),"data_end":str(dates[-1].date()),"first_sample":int(fdate[0]),"last_sample":int(fdate[-1]),"n_dates":len(sidx),"n_stocks":len(codes),"n_rows":N,"sample_every":a.every,"train_label_end":"20230630","val_feature_start":"20230703","val_label_end":"20240628","test_feature_start":"20240701","test_label_end":"20260630","note":"15m built from server 1m; 2026-07 excluded as incomplete"}
    (out/"prepare_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)); print(json.dumps(info,ensure_ascii=False,indent=2))

class DailyGRU(nn.Module):
    def __init__(self):
        super().__init__(); self.gru=nn.GRU(6,30,batch_first=True); self.bn=nn.BatchNorm1d(30); self.fc=nn.Linear(30,1)
    def forward(self,x15,xd=None):
        z,_=self.gru(xd); return self.fc(self.bn(z[:,-1])).squeeze(-1)

class GRU15(nn.Module):
    def __init__(self):
        super().__init__(); self.gru=nn.GRU(6,30,batch_first=True); self.bn=nn.BatchNorm1d(30); self.fc=nn.Linear(30,1)
    def forward(self,x15,xd=None):
        z,_=self.gru(x15); return self.fc(self.bn(z[:,-1])).squeeze(-1)

class Mixed(nn.Module):
    def __init__(self,incremental=False):
        super().__init__(); self.incremental=incremental
        self.gd=nn.GRU(6,30,batch_first=True); self.gm=nn.GRU(6,30,batch_first=True)
        self.bnd=nn.BatchNorm1d(30); self.bnm=nn.BatchNorm1d(30); self.fcd=nn.Linear(30,1); self.fcm=nn.Linear(30,1)
        if not incremental: self.fc=nn.Linear(60,1)
    def daily_out(self,xd):
        z,_=self.gd(xd); return self.fcd(self.bnd(z[:,-1])).squeeze(-1)
    def minute_out(self,x15):
        z,_=self.gm(x15); return self.fcm(self.bnm(z[:,-1])).squeeze(-1)
    def forward(self,x15,xd=None):
        zd,_=self.gd(xd); zm,_=self.gm(x15); d=self.bnd(zd[:,-1]); m=self.bnm(zm[:,-1])
        return (self.fcd(d)+self.fcm(m)).squeeze(-1) if self.incremental else self.fc(torch.cat([d,m],1)).squeeze(-1)
    def stage1(self,xd): return self.daily_out(xd)

def corr_loss(pred,y):
    if pred.shape != y.shape: raise ValueError(f"shape mismatch {pred.shape} vs {y.shape}")
    pc=pred-pred.mean(); yc=y-y.mean(); return -(pc*yc).sum()/(torch.sqrt((pc*pc).sum()*(yc*yc).sum())+1e-8)

def split_indices(f,l):
    return (l<=20230630),(f>=20230703)&(l<=20240628),(f>=20240701)&(l<=20260630)

def batches(idx,dates,batch=512):
    for d in sorted(np.unique(dates[idx])):
        z=idx[dates[idx]==d]
        for j in range(0,len(z),batch): yield z[j:j+batch]

def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); out=Path(a.out); z=np.load(out/"prepared.npz",mmap_mode="r")
    x15,xd,y=z["x15"],z["xd"],z["y"]; f,l=z["feature_date"],z["label_end"]; vm=z["valid"]; tr,va,te=split_indices(f,l); tr&=vm; va&=vm; te&=vm
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=DailyGRU() if a.model=="daily" else GRU15() if a.model=="gru15" else Mixed(a.model=="incremental")
    model.to(dev); ti=np.where(tr)[0]; vi=np.where(va)[0]; best=-1e9; bad=0; hist=[]
    def run_eval(ii):
        model.eval(); vals=[]
        with torch.no_grad():
            for q in batches(ii,f):
                xx=torch.from_numpy(np.asarray(x15[q])).to(dev); dd=torch.from_numpy(np.asarray(xd[q])).to(dev); yy=torch.from_numpy(np.asarray(y[q])).to(dev)
                vals.append(float((-corr_loss(model(xx,dd),yy)).cpu()))
        return float(np.nanmean(vals)) if vals else float("nan")
    if a.model=="incremental":
        opt=torch.optim.Adam(list(model.gd.parameters())+list(model.bnd.parameters())+list(model.fcd.parameters()),lr=1e-3)
        for ep in range(a.epochs):
            model.train(); losses=[]
            for q in batches(ti,f):
                dd=torch.from_numpy(np.asarray(xd[q])).to(dev); yy=torch.from_numpy(np.asarray(y[q])).to(dev); opt.zero_grad(set_to_none=True)
                loss=corr_loss(model.stage1(dd),yy); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
            score=run_eval(vi); hist.append({"stage":"daily","epoch":ep+1,"train_neg_ic":float(np.mean(losses)),"val_ic":score}); print(hist[-1],flush=True)
            if score>best: best=score; bad=0; torch.save(model.state_dict(),out/"incremental_stage1.pt")
            else: bad+=1
            if bad>=a.patience: break
        model.load_state_dict(torch.load(out/"incremental_stage1.pt",map_location=dev,weights_only=True))
        frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(("gd.","bnd.","fcd."))}
        for par in list(model.gd.parameters())+list(model.bnd.parameters())+list(model.fcd.parameters()): par.requires_grad=False
        model.gd.eval(); model.bnd.eval(); model.fcd.eval(); opt=torch.optim.Adam(list(model.gm.parameters())+list(model.bnm.parameters())+list(model.fcm.parameters()),lr=1e-3); best=-1e9; bad=0
        for ep in range(a.epochs):
            model.gm.train(); model.bnm.train(); model.fcm.train(); losses=[]
            for q in batches(ti,f):
                xx=torch.from_numpy(np.asarray(x15[q])).to(dev); dd=torch.from_numpy(np.asarray(xd[q])).to(dev); yy=torch.from_numpy(np.asarray(y[q])).to(dev); opt.zero_grad(set_to_none=True)
                with torch.no_grad(): y1=model.stage1(dd)
                loss=corr_loss(y1.detach()+model.minute_out(xx),yy); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
            score=run_eval(vi); hist.append({"stage":"minute_residual","epoch":ep+1,"train_neg_ic":float(np.mean(losses)),"val_ic":score}); print(hist[-1],flush=True)
            if score>best: best=score; bad=0; torch.save(model.state_dict(),out/f"{a.model}.pt")
            else: bad+=1
            if bad>=a.patience: break
        model.load_state_dict(torch.load(out/f"{a.model}.pt",map_location=dev,weights_only=True))
        diffs=[float((v.detach().cpu()-frozen[k]).abs().max()) for k,v in model.state_dict().items() if k in frozen]; print({"stage1_state_max_abs_diff":max(diffs or [0.0])},flush=True)
    else:
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        for ep in range(a.epochs):
            model.train(); losses=[]
            for q in batches(ti,f):
                xx=torch.from_numpy(np.asarray(x15[q])).to(dev); dd=torch.from_numpy(np.asarray(xd[q])).to(dev); yy=torch.from_numpy(np.asarray(y[q])).to(dev)
                opt.zero_grad(set_to_none=True); loss=corr_loss(model(xx,dd),yy); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); losses.append(float(loss.detach().cpu()))
            score=run_eval(vi); hist.append({"epoch":ep+1,"train_neg_ic":float(np.mean(losses)),"val_ic":score}); print(hist[-1],flush=True)
            if score>best: best=score; bad=0; torch.save(model.state_dict(),out/f"{a.model}.pt")
            else: bad+=1
            if bad>=a.patience: break
    model.load_state_dict(torch.load(out/f"{a.model}.pt",map_location=dev,weights_only=True)); model.eval(); pred=np.full(len(y),np.nan,np.float32)
    with torch.no_grad():
        for q in batches(np.where(te)[0],f):
            xx=torch.from_numpy(np.asarray(x15[q])).to(dev); dd=torch.from_numpy(np.asarray(xd[q])).to(dev); pred[q]=(model.stage1(dd)+model.minute_out(xx)).cpu().numpy() if a.model=="incremental" else model(xx,dd).cpu().numpy()
    np.savez(out/f"{a.model}_pred.npz",pred=pred,feature_date=f,label_end=l,ret=z["ret"],valid=z["valid"],stock=z["stock"]); (out/f"{a.model}_history.json").write_text(json.dumps(hist,indent=2))
    print(json.dumps({"model":a.model,"device":str(dev),"train_rows":int(tr.sum()),"val_rows":int(va.sum()),"test_rows":int(te.sum()),"best_val_ic":best,"epochs":len(hist)},indent=2))

def evaluate(a):
    out=Path(a.out); z=np.load(out/f"{a.model}_pred.npz"); p=z["pred"]; r=z["ret"]; f=z["feature_date"]; l=z["label_end"]; vm=z["valid"]; _,_,te=split_indices(f,l); te&=vm
    dates=np.unique(f[te]); ic=[]; top=[]; n=[]
    for d in dates:
        q=np.where(te&(f==d)&np.isfinite(p))[0]
        if len(q)<10: continue
        ic.append(float(spearmanr(p[q],r[q]).statistic)); k=max(1,len(q)//10); order=q[np.argsort(p[q])[-k:]]; top.append(float(np.nanmean(r[order]))); n.append(len(q))
    mt=float(np.nanmean(top)); ann=(1+mt)**(252/10)-1 if mt>-1 else -1
    res={"model":a.model,"test_feature_first":int(dates[0]) if len(dates) else None,"test_feature_last":int(dates[-1]) if len(dates) else None,"dates":len(ic),"mean_rank_ic":float(np.nanmean(ic)),"rank_ic_std":float(np.nanstd(ic)),"ic_ir":float(np.nanmean(ic)/(np.nanstd(ic)+1e-8)),"top_decile_mean_10d_return":mt,"top_decile_annualized_proxy":ann,"median_cross_section":float(np.median(n)),"assumptions":["single seed","fixed train/validation split","top decile equal weight","no costs/industry-size neutralization; required historical fields unavailable"]}
    (out/f"{a.model}_evaluation.json").write_text(json.dumps(res,ensure_ascii=False,indent=2)); print(json.dumps(res,ensure_ascii=False,indent=2))

def main():
    a=parse()
    if a.cmd=="metadata": metadata(a.data_root)
    elif a.cmd=="prepare": prepare(a)
    elif a.cmd=="train": train(a)
    else: evaluate(a)
if __name__=="__main__": main()
