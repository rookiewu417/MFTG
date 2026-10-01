from pathlib import Path
import argparse,json,time,hashlib
import numpy as np, torch
import reproduce
from fullbase_dataset import FullBase

# early windows (same cutoffs as v2/colab/train.py) -- the formal reproduction only ran 2024H2-2026H1
BLOCKS=[('2022H2',20220701,20221231,20210701,20220630,20210630),('2023H1',20230101,20230630,20220101,20221231,20211231),('2023H2',20230701,20231231,20220701,20230630,20220630),('2024H1',20240101,20240630,20230101,20231231,20221231)]

def model_for(name):
    return reproduce.DailyGRU() if name=='daily' else reproduce.GRU15() if name=='gru15' else reproduce.Mixed(name=='incremental')

def get_day(ds,di,name,cache_dir=None,stage=None,ram_cache=False,cache_backend='npz'):
    need15=(name!='daily' and stage!='daily'); needd=(name!='gru15')
    if cache_dir:
        if cache_backend=='npz':
            q=normalized_cache.load_day(cache_dir,int(ds.date_int[di]),need_x15=need15,need_xd=needd,ram_cache=ram_cache)
        elif cache_backend=='mmap':
            q=stored_npz_loader.load_day(cache_dir,int(ds.date_int[di]),need_x15=need15,need_xd=needd)
        else:
            raise ValueError(cache_backend)
        q['date']=q.pop('feature_date'); q['di']=di; q['valid']=np.ones(len(q['y']),dtype=bool)
        if 'x15' not in q: q['x15']=np.empty((len(q['y']),0,6),np.float32)
        if 'xd' not in q: q['xd']=np.empty((len(q['y']),0,6),np.float32)
        return q
    return ds.day_batch(di,need_x15=need15,need_xd=needd)

def pred(model,name,q,dev):
    xx=torch.from_numpy(q['x15']).to(dev); dd=torch.from_numpy(q['xd']).to(dev)
    with torch.no_grad():
        if name=='incremental': out=model.stage1(dd)+model.minute_out(xx)
        else: out=model(xx,dd)
    return out.detach().cpu().numpy()
def fit_one(model,name,days,eval_days,ds,dev,epochs,patience,cache_dir=None,ram_cache=False,cache_backend='npz'):
    if not days: return []
    def run_eval(stage=None):
        vals=[]
        model.eval()
        for di in eval_days:
            q=get_day(ds,di,name,cache_dir,stage=stage,ram_cache=ram_cache,cache_backend=cache_backend)
            if len(q['y'])>=2:
                with torch.no_grad():
                    xx=torch.from_numpy(q['x15']).to(dev); dd=torch.from_numpy(q['xd']).to(dev); yy=torch.from_numpy(q['y']).to(dev)
                    if name=='incremental' and stage=='daily': pp=model.stage1(dd)
                    elif name=='incremental': pp=model.stage1(dd)+model.minute_out(xx)
                    else: pp=model(xx,dd)
                    vals.append(float((-reproduce.corr_loss(pp,yy)).cpu()))
        return float(np.mean(vals)) if vals else float('nan')
    if name=='incremental':
        opt=torch.optim.Adam(list(model.gd.parameters())+list(model.bnd.parameters())+list(model.fcd.parameters()),lr=1e-3); best=-1e9; beststate=None; bad=0
        for ep in range(epochs):
            model.train(); losses=[]
            for di in days:
                q=get_day(ds,di,name,cache_dir,stage='daily',ram_cache=ram_cache,cache_backend=cache_backend); 
                if len(q['y'])<2: continue
                dd=torch.from_numpy(q['xd']).to(dev); yy=torch.from_numpy(q['y']).to(dev); opt.zero_grad(set_to_none=True); loss=reproduce.corr_loss(model.stage1(dd),yy); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
            score=run_eval(stage='daily'); print({'stage':'daily','epoch':ep+1,'loss':float(np.mean(losses)) if losses else None,'val_ic':score},flush=True)
            if score>best: best=score; beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
            else: bad+=1
            if bad>=patience: break
        model.load_state_dict(beststate)
        frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(('gd.','bnd.','fcd.'))}
        for p in list(model.gd.parameters())+list(model.bnd.parameters())+list(model.fcd.parameters()): p.requires_grad=False
        model.gd.eval(); model.bnd.eval(); model.fcd.eval(); opt=torch.optim.Adam(list(model.gm.parameters())+list(model.bnm.parameters())+list(model.fcm.parameters()),lr=1e-3); best=-1e9; beststate=None; bad=0
        for ep in range(epochs):
            model.gm.train(); model.bnm.train(); model.fcm.train(); losses=[]
            for di in days:
                q=get_day(ds,di,name,cache_dir,stage='minute',ram_cache=ram_cache,cache_backend=cache_backend); 
                if len(q['y'])<2: continue
                xx=torch.from_numpy(q['x15']).to(dev); dd=torch.from_numpy(q['xd']).to(dev); yy=torch.from_numpy(q['y']).to(dev); opt.zero_grad(set_to_none=True)
                with torch.no_grad(): y1=model.stage1(dd)
                loss=reproduce.corr_loss(y1.detach()+model.minute_out(xx),yy); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
            score=run_eval(stage='minute'); print({'stage':'minute_residual','epoch':ep+1,'loss':float(np.mean(losses)) if losses else None,'val_ic':score},flush=True)
            if score>best: best=score; beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
            else: bad+=1
            if bad>=patience: break
        model.load_state_dict(beststate); model.eval(); diff=max(float((v.detach().cpu()-frozen[k]).abs().max()) for k,v in model.state_dict().items() if k in frozen); print({'stage1_state_max_abs_diff':diff},flush=True); return [best]
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); best=-1e9; beststate=None; hist=[]; bad=0
    for ep in range(epochs):
        model.train(); losses=[]
        for di in days:
            q=get_day(ds,di,name,cache_dir)
            if len(q['y'])<2: continue
            xx=torch.from_numpy(q['x15']).to(dev); dd=torch.from_numpy(q['xd']).to(dev); yy=torch.from_numpy(q['y']).to(dev); opt.zero_grad(set_to_none=True); loss=reproduce.corr_loss(model(xx,dd),yy); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); losses.append(float(loss.detach().cpu()))
        score=run_eval(); rec={'epoch':ep+1,'loss':float(np.mean(losses)) if losses else None,'val_ic':score}; hist.append(rec); print(rec,flush=True)
        if score>best: best=score; beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=patience: break
    if beststate is not None: model.load_state_dict(beststate)
    return hist

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-root',default='/root/autodl-tmp/quant/data/ashare'); p.add_argument('--base-dir',default='/root/autodl-tmp/quant/repro_ml_factor_luna_20260917/full_base'); p.add_argument('--out',default='/root/autodl-tmp/quant/repro_ml_factor_luna_20260917/full_rolling'); p.add_argument('--model',choices=['daily','gru15','mixed','incremental'],default='daily'); p.add_argument('--every',type=int,default=5); p.add_argument('--epochs',type=int,default=100); p.add_argument('--patience',type=int,default=10); p.add_argument('--batch',type=int,default=5000); p.add_argument('--cache-dir'); p.add_argument('--ram-cache',action='store_true'); p.add_argument('--cache-backend',choices=['npz','mmap'],default='npz'); p.add_argument('--max-train-days',type=int,default=0); p.add_argument('--max-val-days',type=int,default=0); p.add_argument('--max-test-days',type=int,default=0); p.add_argument('--rounds',default='all'); p.add_argument('--seed',type=int,default=7); a=p.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); ds=FullBase(a.data_root,a.base_dir,a.every); universe_hash=hashlib.sha256(('\n'.join(map(str,ds.codes))).encode()).hexdigest(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); selected=BLOCKS if a.rounds=='all' else [b for b in BLOCKS if b[0] in a.rounds.split(',')]; summary=[]; t0=time.time()
    for name,fs,fe,vs,ve,te in selected:
        train=[di for di in ds.sampled_days() if ds.date_int[di+10]<=te]; val=[di for di in ds.sampled_days() if ds.date_int[di]>=vs and ds.date_int[di+10]<=ve]; test=[di for di in ds.sampled_days() if ds.date_int[di]>=fs and ds.date_int[di]<=fe and ds.date_int[di+10]<=20260630]
        if a.max_train_days: train=train[-a.max_train_days:]
        if a.max_val_days: val=val[:a.max_val_days]
        if a.max_test_days: test=test[:a.max_test_days]
        model=model_for(a.model).to(dev); h=fit_one(model,a.model,train,val,ds,dev,a.epochs,a.patience,a.cache_dir,a.ram_cache,a.cache_backend); model.eval(); rows=[]
        for di in test:
            q=get_day(ds,di,a.model,a.cache_dir,ram_cache=a.ram_cache,cache_backend=a.cache_backend); pp=pred(model,a.model,q,dev); rows.append({'feature_date':q['date'],'label_end':int(ds.date_int[di+10]),'stock':q['stock'].tolist(),'pred':pp.tolist(),'y':q['y'].tolist()})
        ck=out/f'{a.model}_{name}.pt'; torch.save({'model':model.state_dict(),'round':name,'model_name':a.model,'seed':a.seed,'every':a.every,'epochs':a.epochs,'patience':a.patience,'batch':a.batch,'cache_dir':a.cache_dir,'universe_hash':universe_hash,'universe_codes':len(ds.codes),'train_end':te,'val_feature_start':vs,'val_label_end':ve,'test_feature_start':fs,'test_end':fe},ck); (out/f'{a.model}_{name}_pred.json').write_text(json.dumps(rows));
        if rows:
            np.savez(out/f'{a.model}_{name}_pred.npz',pred=np.concatenate([np.asarray(r['pred'],np.float32) for r in rows]),y=np.concatenate([np.asarray(r['y'],np.float32) for r in rows]),stock=np.concatenate([np.asarray(r['stock'],np.int32) for r in rows]),feature_date=np.concatenate([np.full(len(r['pred']),r['feature_date'],np.int32) for r in rows]),label_end=np.concatenate([np.full(len(r['pred']),r['label_end'],np.int32) for r in rows]),valid=np.ones(sum(len(r['pred']) for r in rows),bool),stock_code=np.concatenate([np.asarray(ds.codes,dtype='<U16')[np.asarray(r['stock'],int)] for r in rows]),universe_hash=np.asarray(universe_hash))
        rec={'round':name,'model':a.model,'device':str(dev),'train_days':len(train),'val_days':len(val),'test_days':len(test),'train_val_day_overlap':len(set(train)&set(val)),'checkpoint':str(ck),'history':h,'patience':a.patience,'batch':a.batch}; summary.append(rec); print(rec,flush=True)
    (out/f'{a.model}_rolling_summary.json').write_text(json.dumps({'summary':summary,'seconds':time.time()-t0},indent=2)); print(json.dumps({'device':str(dev),'summary':summary,'seconds':time.time()-t0},indent=2))
if __name__=='__main__': main()
