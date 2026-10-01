"""Risk overlays on the daily active return of a finished portfolio (round 4): scale the active book (blend towards the
benchmark) using only past excess returns. Switching cost = |d scale| * 2 * 10bp (the active leg is traded both ways)."""
import numpy as np
import pandas as pd
import pf

FILES = {'neu opt_sty3': 'general_neu_opt_sty3', 'neu_top opt_sty3': 'general_neu_top_opt_sty3',
         'neu_top ew_h1': 'general_neu_top_ew_h1', 'neu ew_h1': 'general_neu_ew_h1'}


def rule(ex, kind):
    x = ex.to_numpy(); n = len(x); sc = np.ones(n)
    nav = np.cumprod(1 + x)
    for t in range(61, n):
        past = x[:t - 1]                                    # decision for day t uses returns up to t-2 (signal at close t-1)
        if kind == 'dd3':
            p = nav[:t - 1]; dd = p[-1] / p[-60:].max() - 1; sc[t] = 0.5 if dd < -0.03 else 1
        elif kind == 'dd5':
            p = nav[:t - 1]; dd = p[-1] / p[-60:].max() - 1; sc[t] = 0.5 if dd < -0.05 else 1
        elif kind == 'vol':
            v = past[-20:].std() * np.sqrt(252); lt = past[-250:].std() * np.sqrt(252); sc[t] = min(1, lt / max(v, 1e-9))
        elif kind == 'mom':
            sc[t] = 0.5 if past[-20:].sum() < 0 else 1
    cost = np.abs(np.diff(np.r_[1, sc])) * 2 * 1e-3
    return pd.Series(sc * x - cost, index=ex.index)


if __name__ == '__main__':
    rows = []
    for name, f in FILES.items():
        df = pd.read_parquet(f'/home/ivenwu/projects/ml-factor/v2/results/r4_daily/{f}.parquet')
        df = df[df.date <= 20260630].reset_index(drop=True)
        ex = (1 + df.port) / (1 + df.bench_ew) - 1
        for kind in ['none', 'dd3', 'dd5', 'vol', 'mom']:
            e = ex if kind == 'none' else rule(ex, kind)
            nav = np.cumprod(1 + e)
            rows.append({'portfolio': name, 'overlay': kind, 'ann': nav.iloc[-1] ** (252 / len(e)) - 1, 'ir': e.mean() / e.std() * np.sqrt(252),
                         'maxdd': (nav / nav.cummax() - 1).min()})
    out = pd.DataFrame(rows); print(out.round(3).to_string(index=False))
    out.to_csv('/home/ivenwu/projects/ml-factor/v2/results/r4_overlay.csv', index=False)
