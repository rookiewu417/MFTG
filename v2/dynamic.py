"""Regime-adaptive allocation across candidate portfolios using only realised past excess returns.

At rebalance k the weights use periods <= k-2 (period k-1 is not finished when the k-th decision is taken).
Meta excess = sum_c w_c * excess_c - switching cost (|dw| * 20 bp, i.e. a full round trip of the moved capital).
"""
import sys
import numpy as np
import pandas as pd
import robust as rb

CANDS = {'xgb_neu': 'runB/xgb', 'xgb': 'runA_xgb/xgb', 'xgb_top': 'exp_xgb_top/xgb', 'fusion_top': 'exp_fusion_top/fusion'}
rb.CANDS.clear(); rb.CANDS.update(CANDS)


def series(cfg):
    pb = rb.per_block(cfg)
    s = {c: pd.concat([pb[(c, b)] for b, _, _ in rb.BLOCKS if (c, b) in pb]).set_index('date').ex for c in CANDS}
    df = pd.DataFrame(s).dropna()
    blk = pd.Series(index=df.index, dtype=object)
    for b, s_, e in rb.BLOCKS: blk[(df.index >= s_) & (df.index <= e)] = b
    return df, blk


def allocate(df, rule, L):
    X = df.to_numpy(); n, m = X.shape; W = np.full((n, m), 1 / m)
    for k in range(n):
        hist = X[max(0, k - 1 - L):max(0, k - 1)]
        if len(hist) < max(3, L // 2): continue
        mu = hist.mean(0); sd = hist.std(0) + 1e-9
        if rule == 'best1': w = np.eye(m)[mu.argmax()]
        elif rule == 'pos_mean': w = np.clip(mu, 0, None); w = w / w.sum() if w.sum() > 0 else np.full(m, 1 / m)
        elif rule == 'pos_ir': w = np.clip(mu / sd, 0, None); w = w / w.sum() if w.sum() > 0 else np.full(m, 1 / m)
        else: w = np.full(m, 1 / m)
        W[k] = w
    ex = (W * X).sum(1) - 0.002 * np.abs(np.diff(np.vstack([W[:1], W]), axis=0)).sum(1)
    return pd.Series(ex, index=df.index), pd.DataFrame(W, index=df.index, columns=df.columns)


def perf(ex, per_year=252 / 5.5):
    if len(ex) < 8: return {'cum': np.prod(1 + ex) - 1}
    return {'ann': np.prod(1 + ex) ** (per_year / len(ex)) - 1, 'ir': ex.mean() / ex.std() * np.sqrt(per_year)}


if __name__ == '__main__':
    pd.set_option('display.width', 220)
    for cname, cfg in rb.CFGS.items():
        df, blk = series(cfg)
        res = {}
        for c in CANDS: res[c] = df[c]
        for rule in ['equal', 'best1', 'pos_mean', 'pos_ir']:
            for L in ([12] if rule == 'equal' else [6, 12, 24]):
                ex, W = allocate(df, rule, L); res[f'{rule}_L{L}' if rule != 'equal' else 'equal'] = ex
        out = {}
        for k, ex in res.items():
            row = {}
            for label, keep in {'8win': [b for b, _, _ in rb.BLOCKS][:-1], 'main4': ['2024H2', '2025H1', '2025H2', '2026H1'], 'early4': ['2022H2', '2023H1', '2023H2', '2024H1']}.items():
                p = perf(ex[blk.isin(keep).to_numpy()]); row[f'{label}_ann'] = p.get('ann'); row[f'{label}_ir'] = p.get('ir')
            row['july_cum'] = perf(ex[(blk == '2026H2').to_numpy()])['cum']
            out[k] = row
        print(f'\n===== {cname}'); print(pd.DataFrame(out).T.round(3).to_string())
