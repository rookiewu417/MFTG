import glob, os, time, json
import numpy as np

paths = sorted(glob.glob('normalized_cache/days/*.npz'))
sample = np.load(paths[0], allow_pickle=False)
sizes = {k: (v.shape, str(v.dtype), int(v.nbytes)) for k,v in sample.items()}
def bench(mode):
    t=time.perf_counter(); total=0
    if mode == 'ram':
        xs=[np.load(p, allow_pickle=False)['x15'] for p in paths]
        total=sum(x.nbytes for x in xs)
        for x in xs: total += float(x.sum())*0
    elif mode == 'torch':
        import torch
        xs=[torch.from_numpy(np.load(p, allow_pickle=False)['x15']) for p in paths]
        total=sum(x.numel()*x.element_size() for x in xs)
        for x in xs: total += float(x.sum())*0
    else:
        for p in paths:
            x=np.load(p, allow_pickle=False)['x15']; total += x.nbytes; total += float(x.sum())*0
    return {'seconds':time.perf_counter()-t,'bytes':total}
print(json.dumps({'files':len(paths),'sample':sizes,'disk_bytes':sum(os.path.getsize(p) for p in paths),'seq_npz':bench('disk'),'ram':bench('ram'),'torch':bench('torch')}, indent=2))
