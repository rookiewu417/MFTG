"""Select tabular features (selection-period stats only), assemble the upload bundle, and split it for colab upload."""
import json, shutil, subprocess, time
import numpy as np
from common import OUT

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)

MAX_EXT = 220      # Ceres + builtin features kept on top of the own derived ones
MAX_CORR = 0.85
st = json.loads((OUT / 'feature_stats.json').read_text())
F = OUT / 'fac'; P = OUT / 'pack'
if P.exists(): shutil.rmtree(P)
P.mkdir()

ok = {k: v for k, v in st.items() if 'error' not in v and v['n_ic'] >= 300 and v['cov_sel'] >= 0.6 and v['cov_late'] >= 0.6 and np.isfinite(v['icir'])}
derived = sorted(k for k in ok if k.startswith('d_'))
ext = sorted((k for k in ok if not k.startswith('d_')), key=lambda k: -abs(ok[k]['icir']))
log('usable', len(ok), 'derived', len(derived), 'external', len(ext), 'dropped', len(st) - len(ok))

g = np.load(OUT / 'grid.npz')
sample_rows = np.linspace(0, 400, 30).astype(int)      # inside the selection period (first 418 grid rows)
def vec(name):
    a = np.load(F / f'{name}.npy', mmap_mode='r')[sample_rows].astype(np.float32).ravel()
    a = np.where(a == 255, 126.5, a) - 126.5
    return a / (np.linalg.norm(a) + 1e-9)
chosen = list(derived); V = np.stack([vec(k) for k in chosen])
for k in ext:
    if len(chosen) - len(derived) >= MAX_EXT: break
    v = vec(k)
    if np.max(np.abs(V @ v)) < MAX_CORR:
        chosen.append(k); V = np.vstack([V, v])
log('selected', len(chosen), 'external kept', len(chosen) - len(derived))

G = len(g['grid']); N = np.load(F / f'{chosen[0]}.npy', mmap_mode='r').shape[1]
tab = np.empty((G, N, len(chosen)), np.uint8)
for j, k in enumerate(chosen): tab[:, :, j] = np.load(F / f'{k}.npy')
np.save(P / 'tab.npy', tab); del tab
(P / 'tab_cols.json').write_text(json.dumps(chosen))
(OUT / 'selected_features.json').write_text(json.dumps({k: ok[k] for k in chosen}, indent=1))
for f in ['univ', 'y10', 'y5', 'industry', 'log_circ_mv', 'seq']:
    shutil.copy(OUT / 'base' / f'{f}.npy', P / f'{f}.npy')
shutil.copy(OUT / 'axes.npz', P / 'axes.npz'); shutil.copy(OUT / 'grid.npz', P / 'grid.npz')
shutil.copy(OUT / 'base' / 'meta.json', P / 'meta.json')
log('pack written', subprocess.run(['du', '-sh', str(P)], capture_output=True, text=True).stdout.strip())

U = OUT / 'upload'
if U.exists(): shutil.rmtree(U)
U.mkdir()
subprocess.run(f"tar -C {OUT} -cf - pack | pigz -3 -p 12 | split -b 60m -d -a 3 - {U}/pack.tgz.part", shell=True, check=True)
log('upload parts', subprocess.run(f'ls {U} | wc -l; du -sh {U}', shell=True, capture_output=True, text=True).stdout.split())
