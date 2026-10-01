"""Final tradability profile of the recommended long-only blend over 2022H2-2026H1 (+ July 2026 holdout)."""
from dataclasses import replace
import numpy as np
import pandas as pd
import backtest as bt

R = '/home/ivenwu/projects/ml-factor/v2/runs'
B9 = ['2022H2', '2023H1', '2023H2', '2024H1', '2024H2', '2025H1', '2025H2', '2026H1', '2026H2']
MODELS = {'neuX_topX (xgb_neu + xgb_top)': 'ens/neuX_topX', 'xgb_neu': 'runB/xgb', 'xgb_top': 'exp_xgb_top/xgb'}
base = replace(bt.Cfg(), ema_hl=3, q_out=0.2)
VAR = {'ema3+buf20 (open, 10bp)': base, 'plain': bt.Cfg(), '+ vwap': replace(base, exec_px='vwap'),
       '+ adv>=100m': replace(base, min_adv=1e8), 'AUM 5e8': replace(base, aum=5e8), 'AUM 10e8': replace(base, aum=1e9),
       'AUM 30e8': replace(base, aum=3e9)}
rows = []
for m, p in MODELS.items():
    df = bt.load_scores({b: f'{R}/{p}/{b}_pred.npz' for b in B9})
    panels = {}
    for v, cfg in VAR.items():
        if cfg.ema_hl not in panels: panels[cfg.ema_hl] = bt.score_panel(df, cfg.ema_hl)
        S = panels[cfg.ema_hl]
        r = bt.simulate(S, bt.rebal_dates(S, 20220701, 20260731, cfg.hold), cfg)
        main = r[r.date <= 20260630]; july = r[r.date > 20260630]
        s = bt.summary(main, cfg.hold)
        yearly = main.assign(y=main.date // 10000).groupby('y').ex.apply(lambda x: np.prod(1 + x) - 1)
        rows.append({'model': m, 'variant': v, 'excess_ann': s['excess_ann'], 'ir': s['excess_ir'], 'maxdd': s['maxdd'],
                     'turnover': s['turnover'], 'cost_ann': s['cost_ann'], 'july_cum': np.prod(1 + july.ex) - 1,
                     **{f'y{y}': x for y, x in yearly.items()}})
        print(m, v, {k: round(float(x), 3) for k, x in rows[-1].items() if k not in ('model', 'variant')}, flush=True)
out = pd.DataFrame(rows)
out.to_csv('/home/ivenwu/projects/ml-factor/v2/results/final_topdecile.csv', index=False)
