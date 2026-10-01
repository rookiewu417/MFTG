"""Round 4: portfolio construction / tradability / risk grid on the daily simulator (pf.py).

usage: pixi run python round4.py GROUP [GROUP ...]   (groups below; results -> results/r4_<group>.csv)
All runs: 2022-07-01 .. 2026-07-31; stats on 2022-07..2026-06, 'july' = 2026-07 holdout.
"""
import sys
import multiprocessing as mp
from dataclasses import replace
import numpy as np
import pandas as pd
import pf

SIG = {'neu': ['runB/xgb'], 'neu_top': ['runB/xgb', 'exp_xgb_top/xgb']}
SIG_ALL = {**SIG, 'xgb': ['runA_xgb/xgb'], 'xgb_top': ['exp_xgb_top/xgb'], 'fusion_top': ['exp_fusion_top/fusion'],
           'mlp': ['runA/mlp'], 'fusion_neu': ['fusion_neu/fusion'], 'neu_fus_top': ['runB/xgb', 'exp_fusion_top/fusion']}
if len(sys.argv) > 2 and sys.argv[1] == '--sig': SIG = {k: v for k, v in SIG_ALL.items() if k in sys.argv[2].split(',')}; del sys.argv[1:3]
B = pf.PCfg()
STY3 = (('size', 0.3), ('beta', 0.3), ('volatility', 0.3))
OPT = replace(B, weight='opt', max_w=0.005, tc_pen=0.5)
OPTB = replace(OPT, hold=1, sty_caps=tuple((k, 0.05) for k, _ in STY3))   # round-4 default
EWB = replace(B, hold=1)


def grid(group):
    g = {}
    if group == 'sched':
        for h, K in [(5, 1), (5, 5), (1, 1), (2, 1), (10, 2), (10, 1), (20, 4)]:
            g[f'hold{h}_K{K}'] = replace(B, hold=h, tranches=K)
    elif group == 'buffer':
        for ema in [0, 2, 3, 5, 8]:
            for qo in [0.1, 0.2, 0.3]:
                g[f'ema{ema}_qo{qo}'] = replace(B, ema_hl=ema, q_out=qo)
        for q, qo in [(0.05, 0.1), (0.05, 0.15), (0.15, 0.3), (0.2, 0.3), (0.2, 0.4)]:
            g[f'q{q}_qo{qo}'] = replace(B, q=q, q_out=qo)
    elif group == 'weight':
        for w in ['ew', 'score', 'ivol']:
            for mw in [0.005, 0.01]:
                g[f'{w}_mw{mw}'] = replace(B, weight=w, max_w=mw)
        for a in [3e7, 1e8]:
            g[f'ew_adv{a:.0e}'] = replace(B, min_adv=a)
    elif group == 'opt':
        for mw in [0.003, 0.005, 0.01]:
            for tc in [0.0, 0.5, 1.5]:
                g[f'opt_mw{mw}_tc{tc}'] = replace(OPT, max_w=mw, tc_pen=tc)
        for ic in [0.01, 0.03, 0.06]:
            g[f'opt_ind{ic}'] = replace(OPT, ind_cap=ic)
        for pool in [0.2, 0.5]:
            g[f'opt_pool{pool}'] = replace(OPT, opt_pool=pool)
        g['opt_ema0'] = replace(OPT, ema_hl=0)
        g['opt_hold1'] = replace(OPT, hold=1, tc_pen=1.5)
    elif group == 'risk':
        for cap in [0.1, 0.3, 0.6]:
            g[f'opt_sty3_{cap}'] = replace(OPT, sty_caps=tuple((k, cap) for k, _ in STY3))
        for k in ['size', 'beta', 'volatility', 'momentum', 'liquidity', 'value', 'reversal']:
            g[f'opt_{k}0.2'] = replace(OPT, sty_caps=((k, 0.2),))
        g['opt_all7_0.3'] = replace(OPT, sty_caps=tuple((k, 0.3) for k in ['size', 'beta', 'volatility', 'momentum', 'liquidity', 'value', 'reversal']))
    elif group == 'opt2':
        for cap in [0.0, 0.05, 0.1]:
            for tc in [0.25, 0.5]:
                for h in [1, 5]:
                    g[f'sty3_{cap}_tc{tc}_h{h}'] = replace(OPT, sty_caps=tuple((k, cap) for k, _ in STY3), tc_pen=tc, hold=h)
        g['all7_0.1'] = replace(OPT, sty_caps=tuple((k, 0.1) for k in ['size', 'beta', 'volatility', 'momentum', 'liquidity', 'value', 'reversal']))
        g['sty3_0.1_ind0.01'] = replace(OPT, sty_caps=tuple((k, 0.1) for k, _ in STY3), ind_cap=0.01)
        g['sty3_0.1_mw0.003'] = replace(OPT, sty_caps=tuple((k, 0.1) for k, _ in STY3), max_w=0.003)
        g['sty3_0.1_mw0.01'] = replace(OPT, sty_caps=tuple((k, 0.1) for k, _ in STY3), max_w=0.01)
        g['sty3_0.1_ema0'] = replace(OPT, sty_caps=tuple((k, 0.1) for k, _ in STY3), ema_hl=0)
        g['sty3_0.1_ema5'] = replace(OPT, sty_caps=tuple((k, 0.1) for k, _ in STY3), ema_hl=5)
        g['sv_0.1'] = replace(OPT, sty_caps=(('size', 0.1), ('volatility', 0.1)))
        g['sb_0.1'] = replace(OPT, sty_caps=(('size', 0.1), ('beta', 0.1)))
    elif group == 'capacity':
        for aum in [5e8, 1e9, 3e9, 1e10]:
            g[f'ew_aum{aum:.0e}'] = replace(EWB, aum=aum)
            g[f'ew_aum{aum:.0e}_cap'] = replace(EWB, aum=aum, pos_adv=0.5, part=0.1)
            g[f'opt_aum{aum:.0e}'] = replace(OPTB, aum=aum)
            g[f'opt_aum{aum:.0e}_cap'] = replace(OPTB, aum=aum, pos_adv=0.5, part=0.1)
            g[f'opt_aum{aum:.0e}_cap_tc1'] = replace(OPTB, aum=aum, pos_adv=0.5, part=0.1, tc_pen=1.0)
            g[f'opt_aum{aum:.0e}_cap_mw01'] = replace(OPTB, aum=aum, pos_adv=0.5, part=0.1, max_w=0.01)
    elif group == 'hedge':
        for u in ['all', 'csi1800', 'csi1000', 'csi500']:
            g[f'ew_{u}'] = replace(EWB, universe=u)
            g[f'opt_{u}'] = replace(OPTB, universe=u, max_w=0.01 if u != 'all' else 0.005)
    elif group == 'cap2':
        for aum in [5e8, 1e9, 3e9, 1e10]:
            for ip in [0.5, 1.0, 2.0]:
                g[f'opt_aum{aum:.0e}_imp{ip}'] = replace(OPTB, aum=aum, pos_adv=0.5, part=0.1, impact_pen=ip)
            g[f'opt_aum{aum:.0e}_imp1_h5'] = replace(OPTB, aum=aum, pos_adv=0.5, part=0.1, impact_pen=1.0, hold=5)
    elif group == 'enh':
        for idx, floor, szc in [('csi1000', 0.8, 0.2), ('csi500', 0.6, 0.05)]:
            E = replace(OPTB, bench_idx=idx, member_floor=floor, max_w=0.008, ind_cap=0.02, cost_bp=20, exec_px='vwap',
                        opt_pool=0.3, sty_caps=(('size', szc),))
            for tc in [0.25, 0.5, 1.0]:
                g[f'{idx}_size_tc{tc}'] = replace(E, tc_pen=tc)
                g[f'{idx}_sty3_tc{tc}'] = replace(E, tc_pen=tc, sty_caps=(('size', szc), ('beta', 0.1), ('volatility', 0.1)))
            g[f'{idx}_sty3_tc0.5_h5'] = replace(E, tc_pen=0.5, hold=5, sty_caps=(('size', szc), ('beta', 0.1), ('volatility', 0.1)))
            g[f'{idx}_size_tc0.5_ema0'] = replace(E, tc_pen=0.5, ema_hl=0)
            g[f'{idx}_sty3_tc0.5_ema0'] = replace(E, tc_pen=0.5, ema_hl=0, sty_caps=(('size', szc), ('beta', 0.1), ('volatility', 0.1)))
    elif group == 'general':
        g['ew_h5_round3'] = B
        g['ew_h1'] = EWB
        g['opt_sty3'] = OPTB
        g['opt_sty3_tc1'] = replace(OPTB, tc_pen=1.0)
        E = replace(OPTB, bench_idx='csi1000', member_floor=0.8, max_w=0.008, ind_cap=0.02, cost_bp=20, exec_px='vwap',
                    sty_caps=(('size', 0.2), ('beta', 0.1), ('volatility', 0.1)))
        g['enh1000_sty3'] = E
        g['enh1000_size'] = replace(E, sty_caps=(('size', 0.2),))
    return g


_S = {}
def job(args):
    group, sig, name, cfg = args
    key = (sig, cfg.ema_hl)
    if key not in _S: _S[key] = pf.load_signal(SIG[sig], cfg.ema_hl)
    df = pf.run(_S[key], 20220701, 20260731, cfg)
    out = []
    benches = ['bench_ew', 'csi500', 'csi1000'] if group == 'hedge' else [cfg.bench_idx] if cfg.bench_idx else ['bench_ew']
    for b in benches:
        out.append({'group': group, 'signal': sig, 'config': name, 'bench': b, **pf.report(df, b)})
    df.to_parquet(f'/home/ivenwu/projects/ml-factor/v2/results/r4_daily/{group}_{sig}_{name}.parquet')
    return out


if __name__ == '__main__':
    import os; os.makedirs('/home/ivenwu/projects/ml-factor/v2/results/r4_daily', exist_ok=True)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    for group in sys.argv[1:]:
        tasks = [(group, s, n, c) for s in SIG for n, c in grid(group).items()]
        with mp.get_context('fork').Pool(10) as pool:
            rows = [r for rs in pool.imap_unordered(job, tasks) for r in rs]
        res = pd.DataFrame(rows).sort_values(['bench', 'signal', 'config'])
        res.to_csv(f'/home/ivenwu/projects/ml-factor/v2/results/r4_{group}.csv', index=False)
        cols = ['signal', 'config', 'bench', 'ann', 'ir', 'maxdd', 'to_ann', 'cost_ann', 'n', 'worst_half', 'neg_halves', 'july']
        print(f'===== {group}'); print(res[cols].round(3).to_string(index=False), flush=True)
