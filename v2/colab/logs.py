import os
for t in ['runA_xgb', 'runB_xgb', 'runA_fusion', 'runB_fusion', 'exp']:
    p = f'/content/out/{t}.log'
    if not os.path.exists(p): continue
    lines = [l for l in open(p, 'rb').read().replace(b'\x00', b'').decode(errors='ignore').splitlines() if '==>' in l or 'Error' in l]
    print('##', t); print('\n'.join(lines[-8:]))
for f in ['lane1.done', 'lane2.done']:
    if os.path.exists(f'/content/out/{f}'): print('DONE', f)
