"""Point-in-time rules of the extra inputs (build_extra.asof_panel / event_age) on the real trading calendar."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip('duckdb')
be = pytest.importorskip('build_extra')

D, C = be.dates, be.codes
c0 = C[0]
ti = lambda s: int(D.searchsorted(pd.Timestamp(s)))


def _p(rows, limit=300):
    df = pd.DataFrame(rows, columns=['ts_code', 'ann_date', 'end_date', 'v'])
    return be.asof_panel(df, ['v'], limit)['v'][:, 0]


def test_value_starts_on_the_first_trading_day_on_or_after_the_announcement():
    sat = '2024-03-16'                                            # Saturday
    a = _p([(c0, sat, '2023-12-31', 1.0)])
    t = ti(sat)
    assert D[t] == pd.Timestamp('2024-03-18')
    assert np.isnan(a[t - 1]) and a[t] == 1.0 and a[t + 5] == 1.0


def test_first_disclosure_kept_and_older_periods_never_replace_newer():
    a = _p([(c0, '2024-04-10', '2024-03-31', 1.0),
            (c0, '2024-05-10', '2024-03-31', 9.0),                 # revision of the same period: ignored
            (c0, '2024-06-10', '2023-12-31', 7.0),                 # older period announced later: ignored
            (c0, '2024-08-20', '2024-06-30', 2.0)])
    assert a[ti('2024-04-10')] == 1.0 and a[ti('2024-05-10')] == 1.0 and a[ti('2024-06-10')] == 1.0
    assert a[ti('2024-08-20')] == 2.0


def test_forward_fill_limit():
    a = _p([(c0, '2024-04-10', '2024-03-31', 1.0)], limit=10)
    t = ti('2024-04-10')
    assert a[t + 10] == 1.0 and np.isnan(a[t + 11])


def test_event_age():
    g = be.event_age(pd.DataFrame({'ts_code': [c0, c0], 'ann_date': ['2024-04-10', '2024-04-20']}))[:, 0]
    t1, t2 = ti('2024-04-10'), ti('2024-04-20')
    assert np.isnan(g[t1 - 1]) and g[t1] == 0 and g[t1 + 2] == 2 and g[t2] == 0
