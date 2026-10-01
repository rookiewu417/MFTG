"""Tradability study on the main windows (2024H2-2026H1): turnover control, liquidity, execution, capacity."""
from dataclasses import replace
import pandas as pd
import backtest as bt

BL = ['2024H2', '2025H1', '2025H2', '2026H1']
R = '/home/ivenwu/projects/ml-factor/v2/runs'
MODELS = {'xgb_top': 'exp_xgb_top/xgb', 'xgb': 'runA_xgb/xgb', 'fusion_top': 'exp_fusion_top/fusion'}
base = bt.Cfg()
VARIANTS = {
    'base (5d, top10%, open, 10bp)': base,
    'hold 10d': replace(base, hold=10),
    'hold 20d': replace(base, hold=20),
    'ema hl=2': replace(base, ema_hl=2),
    'ema hl=5': replace(base, ema_hl=5),
    'buffer 20%': replace(base, q_out=0.2),
    'buffer 30%': replace(base, q_out=0.3),
    'ema3 + buffer20%': replace(base, ema_hl=3, q_out=0.2),
    'ema3 + buffer20% + 10d': replace(base, ema_hl=3, q_out=0.2, hold=10),
    'adv >= 50m': replace(base, min_adv=5e7),
    'adv >= 100m': replace(base, min_adv=1e8),
    'adv >= 300m': replace(base, min_adv=3e8),
    'vwap execution': replace(base, exec_px='vwap'),
    'top 5%': replace(base, q=0.05, q_out=0.05),
    'top 2%': replace(base, q=0.02, q_out=0.02),
    'csi1000 only': replace(base, universe='csi1000'),
    'csi500 only': replace(base, universe='csi500'),
}
rows = []
panels = {}
for m, path in MODELS.items():
    df = bt.load_scores({b: f'{R}/{path}/{b}_pred.npz' for b in BL})
    for vname, cfg in VARIANTS.items():
        key = (m, cfg.ema_hl)
        if key not in panels: panels[key] = bt.score_panel(df, cfg.ema_hl)
        S = panels[key]
        r = bt.simulate(S, bt.rebal_dates(S, 20240701, 20260630, cfg.hold), cfg)
        rows.append({'model': m, 'variant': vname, **bt.summary(r, cfg.hold)})
        print(m, vname, {k: round(float(v), 3) for k, v in rows[-1].items() if k not in ('model', 'variant')}, flush=True)
res = pd.DataFrame(rows)
res.to_csv('/home/ivenwu/projects/ml-factor/v2/results/tradability.csv', index=False)

# capacity curve for xgb_top under the chosen turnover-control setting, with and without a liquidity floor
df = bt.load_scores({b: f'{R}/exp_xgb_top/xgb/{b}_pred.npz' for b in BL})
cap = []
for label, cfg0 in {'ema3+buf20': replace(base, ema_hl=3, q_out=0.2), 'ema3+buf20+adv100m': replace(base, ema_hl=3, q_out=0.2, min_adv=1e8),
                    'ema3+buf20+adv100m+top5%': replace(base, ema_hl=3, q_out=0.1, q=0.05, min_adv=1e8)}.items():
    S = bt.score_panel(df, cfg0.ema_hl)
    for aum in [0, 1e8, 5e8, 1e9, 3e9, 1e10]:
        cfg = replace(cfg0, aum=aum)
        r = bt.simulate(S, bt.rebal_dates(S, 20240701, 20260630, cfg.hold), cfg)
        cap.append({'setting': label, 'aum_yi': aum / 1e8, **bt.summary(r, cfg.hold)})
        print('capacity', label, aum / 1e8, round(cap[-1]['excess_ann'], 3), round(cap[-1]['excess_ir'], 2), flush=True)
pd.DataFrame(cap).to_csv('/home/ivenwu/projects/ml-factor/v2/results/capacity.csv', index=False)
