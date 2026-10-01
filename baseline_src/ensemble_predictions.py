import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--out',required=True); p.add_argument('--seeds',default='7,11,42'); a=p.parse_args(); seeds=[x for x in a.seeds.split(',') if x]; out=Path(a.out); out.mkdir(parents=True,exist_ok=True); rows=[]
 for model in ('daily','gru15','mixed','incremental'):
  for rnd in ('2024H2','2025H1','2025H2','2026H1'):
   files=[Path(a.root)/f'{model}_{rnd}_seed{s}'/f'{model}_{rnd}_pred.npz' for s in seeds]
   if not all(f.exists() for f in files): continue
   zs=[np.load(f,allow_pickle=False) for f in files]; keys=[]
   for z in zs:
    if 'stock_code' not in z.files or 'universe_hash' not in z.files: raise ValueError(f'{z.files} missing canonical code/hash')
    keys.append(list(zip(z['feature_date'].astype(int),z['stock_code'].astype(str))))
   if any(k!=keys[0] for k in keys[1:]): raise ValueError(f'coverage mismatch {model} {rnd}: {[len(k) for k in keys]}')
   pred=np.mean(np.stack([z['pred'].astype(np.float32) for z in zs],axis=0),axis=0,dtype=np.float32); base=zs[0]
   if any(not np.array_equal(z['y'],base['y']) for z in zs[1:]): raise ValueError(f'label mismatch {model} {rnd}')
   fn=out/f'{model}_{rnd}_ensemble.npz'; np.savez(fn,pred=pred,y=base['y'],feature_date=base['feature_date'],label_end=base['label_end'],stock_code=base['stock_code'],valid=base['valid'],universe_hash=base['universe_hash'],seed_count=np.asarray(len(zs),np.int32),seeds=np.asarray(seeds))
   rows.append({'model':model,'round':rnd,'seeds':seeds,'rows':len(pred),'path':str(fn)})
 Path(a.out,'ensemble_manifest.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
