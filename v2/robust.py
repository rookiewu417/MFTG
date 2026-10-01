"""Walk-forward model selection across 9 windows and the July-2026 holdout.

Every candidate is run through the same long-only backtest (backtest.py). Walk-forward selection picks, for each
window, the candidate with the best excess IR over *earlier* windows only (the first window uses plain xgb).
"""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import backtest as bt

R = Path('/home/ivenwu/projects/ml-factor/v2/runs')
BLOCKS = [('2022H2', 20220701, 20221231), ('2023H1', 20230101, 20230630), ('2023H2', 20230701, 20231231),
          ('2024H1', 20240101, 20240630), ('2024H2', 20240701, 20241231), ('2025H1', 20250101, 20250630),
          ('2025H2', 20250701, 20251231), ('2026H1', 20260101, 20260630), ('2026H2', 20260701, 20260731)]
CANDS = {'xgb': 'runA_xgb/xgb', 'xgb_top': 'exp_xgb_top/xgb', 'xgb_neu': 'runB/xgb', 'mlp': 'runA/mlp',
         'mlp_top': 'exp_mlp_top/mlp', 'fusion_top': 'exp_fusion_top/fusion'}
ENS = {'ens_neu+top': ['runB/xgb', 'exp_xgb_top/xgb'],
       'ens_neu+top+fus': ['runB/xgb', 'exp_xgb_top/xgb', 'exp_fusion_top/fusion'],
       'ens_xgb3': ['runA_xgb/xgb', 'runB/xgb', 'exp_xgb_top/xgb'],
       'ens_5': ['runA_xgb/xgb', 'runB/xgb', 'exp_xgb_top/xgb', 'runA/mlp', 'exp_fusion_top/fusion']}
CANDS.update(ENS)
CFGS = {'plain': bt.Cfg(), 'ema3+buf20': replace(bt.Cfg(), ema_hl=3, q_out=0.2)}
HOLDOUT = '2026H2'


def per_block(cfg):
    out = {}
    for c, path in CANDS.items():
        paths = path if isinstance(path, list) else [path]
        blocks_ok = [b for b, _, _ in BLOCKS if all((R / p / f'{b}_pred.npz').exists() for p in paths)]
        frames = [bt.load_scores({b: str(R / p / f'{b}_pred.npz') for b in blocks_ok}).set_index(['fd', 'j']).pred for p in paths]
        df = pd.concat(frames, axis=1).dropna().mean(1).rename('pred').reset_index()
        files = dict.fromkeys(blocks_ok)
        S = bt.score_panel(df, cfg.ema_hl)
        for b, s, e in BLOCKS:
            if b not in files: continue
            reb = bt.rebal_dates(S, s, e, cfg.hold)
            if len(reb) < 3: continue
            r = bt.simulate(S, reb, cfg)
            r['gap'] = np.diff(np.r_[r.t.to_numpy(), r.t.iloc[-1] + cfg.hold])
            out[(c, b)] = r
    return out


def stats(r):
    if len(r) == 0: return {'excess_ann': np.nan, 'ir': np.nan, 'n': 0}
    per_year = 252 / r.gap.mean(); ex = r.ex
    if len(ex) < 8:   # too short to annualise (holdout): report the cumulative excess instead
        return {'excess_ann': np.nan, 'cum_excess': np.prod(1 + ex) - 1, 'ir': np.nan, 'n': len(ex)}
    return {'excess_ann': np.prod(1 + ex) ** (per_year / len(ex)) - 1, 'ir': ex.mean() / ex.std() * np.sqrt(per_year) if len(ex) > 2 else np.nan,
            'n': len(ex)}


if __name__ == '__main__':
    pd.set_option('display.width', 220)
    for cname, cfg in CFGS.items():
        pb = per_block(cfg)
        tab = pd.DataFrame({c: {b: (lambda x: x['excess_ann'] if np.isfinite(x['excess_ann']) else x.get('cum_excess'))(stats(pb[(c, b)]))
                                for b, _, _ in BLOCKS if (c, b) in pb} for c in CANDS})
        irs = pd.DataFrame({c: {b: stats(pb[(c, b)])['ir'] for b, _, _ in BLOCKS if (c, b) in pb} for c in CANDS})
        print(f'\n===== portfolio setting: {cname}\nexcess by window (annualised; holdout 2026H2 = cumulative over July)'); print((tab * 100).round(1).to_string())
        # walk-forward selection on windows before the holdout
        wf, picks = [], {}
        order = [b for b, _, _ in BLOCKS]
        for i, b in enumerate(order):
            past = order[:i]
            if not past: pick = 'xgb'
            else: pick = irs.loc[past].mean().idxmax()
            picks[b] = pick
            if (pick, b) in pb: wf.append(pb[(pick, b)].assign(block=b, pick=pick))
        print('walk-forward picks:', picks)
        wfr = pd.concat(wf)
        for label, keep in {'2024H2-2026H1 (main)': ['2024H2', '2025H1', '2025H2', '2026H1'],
                            '2023H1-2026H1 (all but first & holdout)': order[1:-1], 'holdout 2026-07': [HOLDOUT]}.items():
            row = {}
            sub = wfr[wfr.block.isin(keep)]; row['walk-forward'] = stats(sub)
            for c in ['xgb', 'xgb_neu', 'xgb_top', 'fusion_top'] + list(ENS):
                parts = [pb[(c, b)] for b in keep if (c, b) in pb]
                row[c] = stats(pd.concat(parts)) if parts else {'n': 0}
            print(f'-- {label}'); print(pd.DataFrame(row).T.round(3).to_string())
