"""Rolling-IC weighted ensemble of MFTG member runs (offline; equal weights are the MFTG-Ens-Neu control).

Each member is z-scored per date. On date t a member's weight is the EWMA (half-life --hl trading days) of its daily
RankIC vs y10 over dates whose label is already known (feature index <= t - 11), floored at 0; equal weights until
--warmup such dates exist or when every weight is 0. Writes runs/mftg/<name>/<block>_pred.npz in the usual format.
"""
import argparse, json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared'))
from common import OUT, load_axes, SHARED

RUNS = OUT.parent / 'runs' / 'mftg'
BLOCKS = ['2022H2', '2023H1', '2023H2', '2024H1', '2024H2', '2025H1', '2025H2', '2026H1', '2026H2X']
LAG = 11   # y10 of feature date t is known at the open of t+11


def rank(a): return pd.Series(a).rank().to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', default='Daily-Neu,Daily-Neu-H256,Daily-Neu-CSI800,Full-Neu')
    ap.add_argument('--hl', type=float, default=120); ap.add_argument('--warmup', type=int, default=40)
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    members = a.members.split(','); name = a.name or f'Ens-Neu-ICW{int(a.hl)}'
    dates, codes = load_axes(); dint = dates.strftime('%Y%m%d').astype(int).to_numpy(); tpos = {d: i for i, d in enumerate(dint)}
    cpos = {c: i for i, c in enumerate(codes)}
    univ = np.load(SHARED / 'base' / 'univ.npy'); y10 = np.load(SHARED / 'base' / 'y10.npy')

    frames = []                                      # long frame: block, t, j, one z-scored column per member
    for b in BLOCKS:
        cols = []
        for m in members:
            z = np.load(RUNS / m / f'{b}_pred.npz', allow_pickle=True)
            d = pd.DataFrame({'t': [tpos[x] for x in z['feature_date']], 'j': [cpos.get(c, -1) for c in z['stock_code']], m: z['pred'].astype(float)})
            d = d[d.j >= 0]; d[m] = d.groupby('t')[m].transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
            cols.append(d.set_index(['t', 'j'])[m])
        f = pd.concat(cols, axis=1).dropna().reset_index(); f['block'] = b; frames.append(f)
    F = pd.concat(frames).sort_values(['t', 'j']).reset_index(drop=True)

    ts = np.sort(F.t.unique()); ic = {}
    for t, g in F.groupby('t'):
        j = g.j.to_numpy(); m = univ[t, j] & np.isfinite(y10[t, j])
        if m.sum() > 100:
            ry = rank(y10[t, j][m]); ic[t] = [np.corrcoef(rank(g[k].to_numpy()[m]), ry)[0, 1] for k in members]
    lam = 0.5 ** (1 / a.hl); W = {}; done = sorted(ic); p = 0; s = np.zeros(len(members)); n = 0
    for t in ts:                                    # fold in every IC whose label is known by t, then fix t's weights
        while p < len(done) and done[p] <= t - LAG:
            s = lam * s + np.asarray(ic[done[p]]); n += 1; p += 1
        w = np.maximum(s, 0)
        W[t] = w / w.sum() if n >= a.warmup and w.sum() > 0 else np.full(len(members), 1 / len(members))
    Wm = np.stack([W[t] for t in F.t]); F['pred'] = (F[members].to_numpy() * Wm).sum(1)

    out = RUNS / name; out.mkdir(parents=True, exist_ok=True)
    for b, g in F.groupby('block'):
        np.savez(out / f'{b}_pred.npz', pred=g.pred.to_numpy(np.float32), feature_date=dint[g.t.to_numpy()].astype(np.int32),
                 stock=g.j.to_numpy(np.int32), stock_code=codes[g.j.to_numpy()].astype('<U12'))
    wt = pd.DataFrame([W[t] for t in ts], index=dint[ts], columns=members)
    (out / 'args.json').write_text(json.dumps({**vars(a), 'members': members}, indent=1))
    wt.to_csv(out / 'weights.csv')
    print(name, 'mean weights by block'); print(wt.groupby(F.drop_duplicates('t').set_index(dint[F.drop_duplicates('t').t])['block']).mean().round(3).to_string())


if __name__ == '__main__':
    main()
