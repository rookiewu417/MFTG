"""Feature-only inference pool builder for existing FullBase checkpoints.

The predictor is injected by the caller so this script does not alter rolling
training.  It never computes valid from y/label_end; those fields are optional
diagnostics only.
"""
import argparse, json
from pathlib import Path
import numpy as np

def build_feature_pool(base_dir, data_root, start, end):
    from fullbase_dataset import FullBase
    ds=FullBase(data_root,base_dir,every=5); rows=[]
    all_days=ds.sampled_days(); wanted=[di for di in all_days if (start is None or ds.date_int[di]>=int(start)) and (end is None or ds.date_int[di]<=int(end))]
    for di in wanted:
        q=ds.day_batch(di,normalize=True,valid_only=False,need_x15=True,need_xd=True)
        feature_valid=np.isfinite(q['x15']).all((1,2)) & np.isfinite(q['xd']).all((1,2))
        for j in np.flatnonzero(feature_valid): rows.append((q['date'],j,ds.codes[j]))
    return rows,ds

def export(src_pred, out, base_dir, data_root, start, end):
    rows,ds=build_feature_pool(base_dir,data_root,start,end); old=np.load(src_pred,allow_pickle=False) if src_pred else None
    fd=np.array([r[0] for r in rows],np.int32); code=np.array([r[2] for r in rows],dtype='U'); pred=np.full(len(rows),np.nan,np.float32)
    if old is not None and 'stock_code' in old.files:
        key={(int(d),str(c)):i for i,(d,c) in enumerate(zip(old['feature_date'],old['stock_code']))}
        for i,(d,c) in enumerate(zip(fd,code)):
            if (int(d),str(c)) in key: pred[i]=old['pred'][key[(int(d),str(c))]]
    np.savez_compressed(out,pred=pred,feature_date=fd,stock_code=code,valid=np.isfinite(pred),feature_valid=np.ones(len(rows),bool))
    report={'feature_rows':len(rows),'predicted_rows':int(np.isfinite(pred).sum()),'missing_checkpoint_predictions':int(np.isnan(pred).sum()),'label_used_for_feature_valid':False,'note':'Inject checkpoint predictor to fill missing rows; existing intersection values are copied only for equality audit.'}
    Path(out).with_suffix('.json').write_text(json.dumps(report,indent=2)); return report

def infer_checkpoint(checkpoint, out, model_name, base_dir, data_root, start, end):
    import torch
    from rolling_full import model_for, pred
    from fullbase_dataset import FullBase
    ds=FullBase(data_root,base_dir,every=5); model=model_for(model_name)
    state=torch.load(checkpoint,map_location='cpu'); model.load_state_dict(state.get('model',state)); model.eval(); rows=[]
    all_days=ds.sampled_days(); wanted=[di for di in all_days if (start is None or ds.date_int[di]>=int(start)) and (end is None or ds.date_int[di]<=int(end))]
    for di in wanted:
        q=ds.day_batch(di,normalize=True,valid_only=False,need_x15=model_name!='daily',need_xd=model_name!='gru15')
        raw15=ds.mm[di-19:di+1]; rawd=ds.daily[di-39:di+1]
        fv=np.isfinite(raw15).all(axis=(0,2,3)) & np.isfinite(rawd).all(axis=(0,2)); ix=np.flatnonzero(fv)
        qq={k:(v[ix] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(fv) else v) for k,v in q.items()}
        pp=pred(model,model_name,qq,torch.device('cpu'))
        for j,v in zip(ix,pp): rows.append((q['date'],ds.codes[j],float(v)))
    np.savez_compressed(out,pred=np.array([r[2] for r in rows],np.float32),feature_date=np.array([r[0] for r in rows],np.int32),stock_code=np.array([r[1] for r in rows],dtype='U'),valid=np.ones(len(rows),bool))
    report={'rows':len(rows),'feature_valid_rows':len(rows),'label_used_for_valid':False,'checkpoint':str(checkpoint),'model':model_name}
    Path(out).with_suffix('.json').write_text(json.dumps(report,indent=2)); return report

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--base-dir',required=True); p.add_argument('--data-root',required=True); p.add_argument('--out',required=True); p.add_argument('--pred'); p.add_argument('--model',default='daily'); p.add_argument('--start',type=int,required=True); p.add_argument('--end',type=int,required=True); p.add_argument('--checkpoint'); a=p.parse_args();
 if not a.checkpoint: raise SystemExit('checkpoint required')
 print(json.dumps(infer_checkpoint(a.checkpoint,a.out,a.model,a.base_dir,a.data_root,a.start,a.end),indent=2))
