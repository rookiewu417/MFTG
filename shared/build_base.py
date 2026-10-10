"""Build the daily panels: axes, universe/tradability masks, labels, sequence features, and eval aux fields.

Everything is point-in-time at the close of feature date t; labels start at the next open (t+1).
Outputs go to build/shared/base/*.npy.
"""
import time
import numpy as np
import pandas as pd
from common import LAKE, OUT, CAL_START, CAL_END, con, to_panel, SHARED

t0 = time.time()
def log(*a): print(f'[{time.time()-t0:6.0f}s]', *a, flush=True)

B = SHARED / 'base'; B.mkdir(parents=True, exist_ok=True)
c = con()

# ---------- axes ----------
cal = c.sql(f"select cal_date from '{LAKE}/meta/trade_cal.parquet' where exchange='SSE' and is_open=1").df()
dates = pd.DatetimeIndex(sorted(pd.to_datetime(cal.cal_date)))
dates = dates[(dates >= CAL_START) & (dates <= CAL_END)]
px = c.sql(f"""select trade_date, ts_code, open, high, low, close, pre_close, vol, amount
               from read_parquet('{LAKE}/daily/price/*/*.parquet', hive_partitioning=false)
               where trade_date between '{CAL_START}' and '{CAL_END}' and ts_code not like '%.BJ'""").df()
have = set(pd.to_datetime(px.trade_date.unique()))
missing = [d for d in dates if d not in have]
assert len(missing) <= 3, f'calendar days without price data: {missing[:10]}'
dates = dates[dates.isin(list(have))]
codes = np.array(sorted(px.ts_code.unique()), dtype=object)
T, N = len(dates), len(codes)
np.savez(SHARED / 'axes.npz', dates=dates.to_numpy(), codes=codes)
log('axes', T, N, dates[0].date(), dates[-1].date())

P = to_panel(px, dates, codes, ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount'])
del px
adj = to_panel(c.sql(f"""select trade_date, ts_code, adj_factor from read_parquet('{LAKE}/daily/adj_factor/*/*.parquet', hive_partitioning=false)
                          where trade_date between '{CAL_START}' and '{CAL_END}'""").df(), dates, codes, ['adj_factor'])['adj_factor']
adj = pd.DataFrame(adj).ffill().bfill().to_numpy(np.float32)   # adj factor only changes on ex-dates
IND = to_panel(c.sql(f"""select trade_date, ts_code, turnover_rate, turnover_rate_f, volume_ratio, pe_ttm, pb, ps_ttm, dv_ttm, total_mv, circ_mv
                          from read_parquet('{LAKE}/daily/indicator/*/*.parquet', hive_partitioning=false)
                          where trade_date between '{CAL_START}' and '{CAL_END}'""").df(), dates, codes,
               ['turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ttm', 'total_mv', 'circ_mv'])
LIM = to_panel(c.sql(f"""select trade_date, ts_code, up_limit, down_limit from read_parquet('{LAKE}/daily/limit/*/*.parquet', hive_partitioning=false)
                          where trade_date between '{CAL_START}' and '{CAL_END}'""").df(), dates, codes, ['up_limit', 'down_limit'])
log('daily panels loaded')

traded = np.isfinite(P['close']) & (np.nan_to_num(P['vol']) > 0) & np.isfinite(P['open']) & (np.nan_to_num(P['open']) > 0)
aopen = np.where(traded, P['open'] * adj, np.nan)
aclose = np.where(traded, P['close'] * adj, np.nan)

# ---------- universe ----------
lst = c.sql(f"select ts_code, list_date, delist_date from '{LAKE}/meta/list.parquet'").df()
ld = pd.Series(pd.to_datetime(lst.list_date).to_numpy(), index=lst.ts_code)
first_seen = np.argmax(traded, axis=0)
list_pos = np.array([dates.searchsorted(ld.get(cd, pd.NaT)) if pd.notna(ld.get(cd, pd.NaT)) else first_seen[j] for j, cd in enumerate(codes)])
age = np.arange(T)[:, None] - list_pos[None, :]          # trading days since listing (negative before listing)
st = np.zeros((T, N), bool)
nm = c.sql(f"select ts_code, name, start_date, end_date, ann_date from '{LAKE}/meta/name.parquet'").df()
nm = nm[nm.name.str.contains('ST|退', regex=True, na=False)]
cidx = pd.Index(codes)
for r in nm.itertuples():
    j = cidx.get_indexer([r.ts_code])[0]
    if j < 0: continue
    # ST is known from max(start_date, ann_date); in the 2026-09 lake ann_date <= start_date always, so this equals start_date
    a = dates.searchsorted(max(pd.Timestamp(r.start_date), pd.Timestamp(r.ann_date) if pd.notna(r.ann_date) else pd.Timestamp(r.start_date)))
    b = T if pd.isna(r.end_date) else dates.searchsorted(pd.Timestamp(r.end_date), side='right')
    st[a:b, j] = True
univ = traded & ~st & (age >= 120)
log('universe mean size', int(univ.sum(1).mean()), 'st share', round(float(st.mean()), 4))

# ---------- entry feasibility and labels ----------
def shift(a, k):
    out = np.full_like(a, np.nan if a.dtype.kind == 'f' else False)
    if k > 0: out[:-k] = a[k:]
    return out

up_hit_open = np.nan_to_num(P['open']) >= np.nan_to_num(LIM['up_limit'], nan=np.inf) - 0.005
buyable = shift(traded & ~up_hit_open, 1).astype(bool)     # can buy at the t+1 open
def fwd(h):
    r = shift(aopen, 1 + h) / shift(aopen, 1) - 1          # open(t+1) -> open(t+1+h)
    return np.where(buyable, r, np.nan).astype(np.float32)
y10, y5, y20 = fwd(10), fwd(5), fwd(20)
y10cc = (shift(aclose, 10) / aclose - 1).astype(np.float32)  # the baseline's close->close label, for comparison only
log('labels: y10 finite on univ', round(float(np.isfinite(y10[univ]).mean()), 4))

# ---------- sequence features (cross-sectionally robust-z per day, fp16) ----------
mf = c.sql(f"""select trade_date, ts_code, net_mf_amount, buy_elg_amount+buy_lg_amount-sell_elg_amount-sell_lg_amount as big_net
               from read_parquet('{LAKE}/flow/moneyflow/*/*/*.parquet', hive_partitioning=false)
               where trade_date between '{CAL_START}' and '{CAL_END}'""").df()
MF = to_panel(mf, dates, codes, ['net_mf_amount', 'big_net']); del mf
iv = c.sql(f"""select * from read_parquet('{LAKE}/features/v1/5min/*/*/*.parquet', hive_partitioning=false)
               where trade_date between '{CAL_START}' and '{CAL_END}'""").df()
IV_COLS = [x for x in iv.columns if x.startswith('i_')]
IV = to_panel(iv, dates, codes, IV_COLS); del iv
log('flow + intraday loaded', len(IV_COLS))

with np.errstate(all='ignore'):
    amt = P['amount']; vwap = P['amount'] / P['vol']
    seq = {
        'r_cc': np.log(P['close'] / P['pre_close']),
        'r_gap': np.log(P['open'] / P['pre_close']),
        'r_hi': np.log(P['high'] / P['close']),
        'r_lo': np.log(P['low'] / P['close']),
        'r_vwap': np.log(vwap / P['close']),
        'l_turn': np.log(IND['turnover_rate_f'] + 1e-3),
        'l_amt': np.log(amt + 1.0),
        'up_hit': (P['close'] >= LIM['up_limit'] - 0.005).astype(np.float32),
        'dn_hit': (P['close'] <= LIM['down_limit'] + 0.005).astype(np.float32),
        'mf_net': MF['net_mf_amount'] / amt,
        'mf_big': MF['big_net'] / amt,
        **IV,
    }
SEQ_COLS = list(seq)
BINARY = {'up_hit', 'dn_hit'}

def robust_z(x, mask):
    x = np.where(mask & np.isfinite(x), x, np.nan)
    med = np.nanmedian(x, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=1, keepdims=True) * 1.4826
    sd = np.nanstd(x, axis=1, keepdims=True)
    scale = np.where(mad > 1e-12, mad, np.where(sd > 1e-12, sd, 1.0))
    return np.clip((x - med) / scale, -5, 5)

S = np.full((T, N, len(SEQ_COLS)), np.nan, np.float16)
for k, name in enumerate(SEQ_COLS):
    x = seq[name]
    S[:, :, k] = np.where(traded, x, np.nan) if name in BINARY else robust_z(x, traded)
del seq, IV, MF
log('seq features', S.shape, 'nan share', round(float(np.isnan(S[univ]).mean()), 4))

# ---------- aux: industry, size, index membership ----------
crit = c.sql(f"select index_code from '{LAKE}/meta/industry_criteria.parquet' where level='L1'").df().index_code.unique()
mem = c.sql(f"select index_code, ts_code, in_date, out_date from '{LAKE}/meta/sw_index_member.parquet'").df()
mem = mem[mem.index_code.isin(crit)]
ind_codes = np.array(sorted(mem.index_code.unique()), dtype=object)
ind = np.full((T, N), -1, np.int16)
for r in mem.itertuples():
    j = cidx.get_indexer([r.ts_code])[0]
    if j < 0: continue
    a = dates.searchsorted(pd.Timestamp(r.in_date))
    b = T if pd.isna(r.out_date) else dates.searchsorted(pd.Timestamp(r.out_date), side='right')
    ind[a:b, j] = np.searchsorted(ind_codes, r.index_code)
log('industry coverage on univ', round(float((ind[univ] >= 0).mean()), 4), 'n_ind', len(ind_codes))

w = c.sql(f"""select trade_date, index_code, ts_code from read_parquet('{LAKE}/index/weight/*/*.parquet', hive_partitioning=false)
              where index_code in ('000300.SH','000905.SH','000852.SH') and trade_date >= '2018-01-01'""").df()
w['trade_date'] = pd.to_datetime(w.trade_date)
bit = {'000300.SH': 1, '000905.SH': 2, '000852.SH': 4}
idx_mem = np.zeros((T, N), np.uint8)
for code, wi in w.groupby('index_code'):   # each index's snapshot applies from the next trading day until its own next snapshot
    snaps = sorted(wi.trade_date.unique())
    for i, s in enumerate(snaps):
        a = dates.searchsorted(s, side='right')
        b = dates.searchsorted(snaps[i + 1], side='right') if i + 1 < len(snaps) else T
        j = cidx.get_indexer(wi.ts_code[wi.trade_date == s]); j = j[j >= 0]
        idx_mem[a:b, j] |= bit[code]
log('index membership sizes (last day)', [int(((idx_mem[-1] & b_) > 0).sum()) for b_ in (1, 2, 4)])

np.save(B / 'univ.npy', univ); np.save(B / 'buyable.npy', buyable); np.save(B / 'traded.npy', traded)
np.save(B / 'y10.npy', y10); np.save(B / 'y5.npy', y5); np.save(B / 'y20.npy', y20); np.save(B / 'y10cc.npy', y10cc)
np.save(B / 'seq.npy', S); np.save(B / 'industry.npy', ind); np.save(B / 'index_member.npy', idx_mem)
np.save(B / 'log_circ_mv.npy', np.log(IND['circ_mv']).astype(np.float32))
np.save(B / 'aopen.npy', aopen.astype(np.float32)); np.save(B / 'aclose.npy', aclose.astype(np.float32))
np.save(B / 'up_limit.npy', LIM['up_limit']); np.save(B / 'down_limit.npy', LIM['down_limit'])
np.save(B / 'open_raw.npy', P['open']); np.save(B / 'close_raw.npy', P['close'])
# raw fields reused by the tabular builder
for k in ['turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ttm', 'total_mv', 'circ_mv']:
    np.save(B / f'ind_{k}.npy', IND[k])
np.save(B / 'amount.npy', P['amount'])
import json
(B / 'meta.json').write_text(json.dumps({'T': T, 'N': N, 'seq_cols': SEQ_COLS, 'ind_codes': list(ind_codes),
                                         'index_bits': bit, 'labels': {'y10': 'open(t+1)->open(t+11), NaN unless buyable at t+1 open',
                                                                        'y5': 'open(t+1)->open(t+6)', 'y20': 'open(t+1)->open(t+21)',
                                                                        'y10cc': 'close(t)->close(t+10), baseline label'}}, ensure_ascii=False, indent=1))
log('done')
