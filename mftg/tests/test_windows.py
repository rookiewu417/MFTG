import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # mftg/
import numpy as np
import pandas as pd
import pytest
from train_mftg import BLOCKS, H, Data, month_blocks

HALF_YEARS = [b[0] for b in BLOCKS if b[0][-2:] in ('H1', 'H2')]
ALL = BLOCKS + [m for n in HALF_YEARS for m in month_blocks(n)]


class FakeData:                     # just the fields Data.split reads
    def __init__(self, train_start=0):
        dates = pd.bdate_range('2018-07-02', '2026-09-10')
        self.dint = dates.strftime('%Y%m%d').astype(int).to_numpy(); self.T = len(dates)
        self.lend = np.array([self.dint[t + H] if t + H < self.T else 99999999 for t in range(self.T)])
        self.L, self.Di, self.train_start = 60, 10, train_start


@pytest.mark.parametrize('blk', ALL, ids=lambda b: b[0])
def test_cutoffs_are_ordered(blk):
    name, fs, fe, vs, ve, te = blk
    assert te < vs <= ve < fs <= fe


@pytest.mark.parametrize('blk', ALL, ids=lambda b: b[0])
@pytest.mark.parametrize('stride', [1, 5])
def test_split_has_no_label_overlap(blk, stride):
    D = FakeData()
    tr, va, tv, test = Data.split(D, blk, stride)
    name, fs, fe, vs, ve, te = blk
    assert len(tr) and len(va) and len(test)
    assert D.lend[tr].max() < D.dint[va].min()           # training labels end before validation features start
    assert D.lend[tv].max() < D.dint[test].min()          # refit (train + val) labels end before the test window
    assert D.dint[test].min() >= fs and D.dint[test].max() <= fe


def test_month_blocks_cover_the_half_year():
    for n in HALF_YEARS:
        b = next(x for x in BLOCKS if x[0] == n); ms = month_blocks(n)
        assert ms[0][1] == b[1] and ms[-1][2] == b[2]
        assert all(prev[2] < nxt[1] for prev, nxt in zip(ms, ms[1:]))

