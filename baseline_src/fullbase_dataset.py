from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
import gru_baseline

class FullBase:
    def __init__(self, data_root='/root/autodl-tmp/quant/data/ashare', base_dir='full_base', every=5):
        self.root=Path(data_root); self.base=Path(base_dir); self.every=every
        info=json.loads((self.base/'meta.json').read_text())
        self.shape=tuple(info['shape']); self.dates=pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(self.root/'daily/post_weighted.parquet',columns=['trade_date']).trade_date.unique()))
        self.dates=self.dates[(self.dates>='2016-01-04')&(self.dates<='2026-06-30')].sort_values()
        sb=pd.read_parquet(self.root/'meta/stock_basic.parquet',columns=['ts_code'])
        dc=pd.read_parquet(self.root/'daily/post_weighted.parquet',columns=['ts_code']).ts_code.dropna().unique()
        self.codes=np.array(sorted(set(sb.ts_code.dropna())|set(dc)),dtype=object)
        if len(self.dates)!=self.shape[0] or len(self.codes)!=self.shape[1]: raise ValueError(f'meta mismatch dates={len(self.dates)} codes={len(self.codes)} shape={self.shape}')
        self.mm=np.memmap(self.base/'minute_base.float32',dtype='float32',mode='r',shape=self.shape)
        self.daily=self._load_daily()
        self.date_int=self.dates.strftime('%Y%m%d').astype(np.int32).to_numpy()
        self.didx={int(x):i for i,x in enumerate(self.date_int)}
    def _load_daily(self):
        cols=['trade_date','ts_code']+gru_baseline.FEATS
        d=pd.read_parquet(self.root/'daily/post_weighted.parquet',columns=cols)
        d.trade_date=pd.to_datetime(d.trade_date)
        d=d[d.trade_date.isin(self.dates)&d.ts_code.isin(self.codes)]
        d=d.drop_duplicates(['trade_date','ts_code'],keep='last')
        out=np.full((len(self.dates),len(self.codes),len(gru_baseline.FEATS)),np.nan,np.float32)
        di=pd.Categorical(d.trade_date,categories=self.dates).codes
        ci=pd.Categorical(d.ts_code,categories=self.codes).codes
        good=(di>=0)&(ci>=0)
        out[di[good],ci[good],:]=d.loc[good,gru_baseline.FEATS].to_numpy(np.float32)
        return out
    def day_batch(self, di, normalize=True, valid_only=True, need_x15=True, need_xd=True):
        lo=di-39; hi=di+11
        if lo<0 or hi>len(self.dates): raise IndexError(di)
        stocks=np.arange(len(self.codes))
        raw15=self.mm[di-19:di+1]
        valid15=np.isfinite(raw15).all(axis=(0,2,3))
        if need_x15:
            x15=raw15.transpose(1,0,2,3).reshape(len(stocks),320,6)
        else:
            x15=np.empty((len(stocks),0,6),np.float32)
        rawd=self.daily[lo:di+1]
        validd=np.isfinite(rawd).all(axis=(0,2))
        xd=rawd.transpose(1,0,2) if need_xd else np.empty((len(stocks),0,6),np.float32)
        y=self.daily[di+10,:,3]/self.daily[di,:,3]-1.0
        valid=valid15&validd&np.isfinite(y)
        if normalize:
            if need_x15: x15=gru_baseline.prep_seq(x15)
            if need_xd: xd=gru_baseline.prep_seq(xd,denom_last=True)
            if need_x15: valid &= np.isfinite(x15).all(axis=(1,2))
            if need_xd: valid &= np.isfinite(xd).all(axis=(1,2))
        if valid_only:
            stocks=stocks[valid]; x15=x15[valid]; xd=xd[valid]; y=y[valid]
        return {'date':int(self.date_int[di]),'di':di,'stock':stocks,'x15':x15,'xd':xd,'y':y,'valid':valid}
    def sampled_days(self, start=None, end=None):
        lo=40; hi=len(self.dates)-11
        if start is not None: lo=max(lo,self.didx.get(int(start),lo))
        if end is not None: hi=min(hi,self.didx.get(int(end),hi)+1)
        return list(range(lo,hi,self.every))

def inspect(a):
    t=time.time(); ds=FullBase(a.data_root,a.base_dir,a.every); days=ds.sampled_days(a.start,a.end)
    q=ds.day_batch(days[0]); print(json.dumps({'shape':ds.shape,'codes':len(ds.codes),'date_count':len(ds.dates),'sampled_days':len(days),'first_date':q['date'],'valid_stocks':len(q['stock']),'x15':list(q['x15'].shape),'xd':list(q['xd'].shape),'y':list(q['y'].shape),'finite':bool(np.isfinite(q['y']).all()),'seconds':time.time()-t},indent=2))

def scan(a):
    ds=FullBase(a.data_root,a.base_dir,a.every); days=ds.sampled_days(a.start,a.end); rows=[]; t=time.time()
    for n,di in enumerate(days):
        q=ds.day_batch(di,normalize=False,valid_only=False); rows.append({'feature_date':q['date'],'valid_stocks':int(q['valid'].sum())})
        if (n+1)%20==0: print('scan',n+1,'/',len(days),flush=True)
    out=Path(a.base_dir)/'fullbase_day_index.json'; out.write_text(json.dumps({'every':a.every,'rows':rows,'total_valid':int(sum(x['valid_stocks'] for x in rows))},indent=2)); print(json.dumps({'days':len(rows),'total_valid':int(sum(x['valid_stocks'] for x in rows)),'out':str(out),'seconds':time.time()-t},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('cmd',choices=['inspect','scan']); p.add_argument('--data-root',default='/root/autodl-tmp/quant/data/ashare'); p.add_argument('--base-dir',default='/root/autodl-tmp/quant/repro_ml_factor_luna_20260917/full_base'); p.add_argument('--every',type=int,default=5); p.add_argument('--start',type=int); p.add_argument('--end',type=int); a=p.parse_args(); inspect(a) if a.cmd=='inspect' else scan(a)
