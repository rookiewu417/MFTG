import time,torch,numpy as np,json
from normalized_cache import load_day
import reproduce
with open('/root/autodl-tmp/quant/repro_ml_factor_luna_20260917/normalized_cache/manifest.json') as f: mm=json.load(f)
date=max(mm['files'],key=lambda d:mm['files'][d]['n'])
t0=time.perf_counter()
for d in list(mm['files'])[:10]: load_day('/root/autodl-tmp/quant/repro_ml_factor_luna_20260917/normalized_cache',d,need_x15=False,need_xd=True)
print('CACHELOAD10_SEC',time.perf_counter()-t0)
q=load_day('/root/autodl-tmp/quant/repro_ml_factor_luna_20260917/normalized_cache',date,need_x15=False,need_xd=True)
print('N',len(q['y']),q['xd'].shape)
dev=torch.device('cuda'); model=reproduce.GRU15().to(dev)
xx=torch.from_numpy(q['xd']).to(dev); yy=torch.from_numpy(q['y']).to(dev)
for _ in range(3):
 model.zero_grad(set_to_none=True); out=model(xx); loss=reproduce.corr_loss(out,yy); loss.backward(); torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats(); model.zero_grad(set_to_none=True); torch.cuda.synchronize()
t=time.perf_counter(); dd=torch.from_numpy(q['xd']).to(dev); torch.cuda.synchronize(); t1=time.perf_counter()
out=model(dd); torch.cuda.synchronize(); t2=time.perf_counter()
loss=reproduce.corr_loss(out,yy); loss.backward(); torch.cuda.synchronize(); t3=time.perf_counter()
opt=torch.optim.Adam(model.parameters(),lr=1e-3); opt.step(); torch.cuda.synchronize(); t4=time.perf_counter()
print('HOSTCOPY',t1-t,'FORWARD',t2-t1,'LOSS_BACKWARD',t3-t2,'OPT',t4-t3,'TOTAL',t4-t,'ALLOC_MB',torch.cuda.memory_allocated()/2**20,'RESERVED_MB',torch.cuda.memory_reserved()/2**20,'PEAK_ALLOC_MB',torch.cuda.max_memory_allocated()/2**20)
