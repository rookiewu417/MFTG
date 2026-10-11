"""--intra_center, --half_life and multi-label runs: unit checks plus a tiny end-to-end CPU run on synthetic data."""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # mftg/
import json, subprocess
import numpy as np
import pandas as pd
import pytest
import torch
import train_mftg
from train_mftg import MFTG, IntraEnc, epoch_days


@pytest.mark.parametrize('center', [1, 2])
def test_centering_removes_the_day_offset(center):
    torch.manual_seed(0)
    enc = IntraEnc(d=16, layers=1, cin=5, center=center).eval()
    x = torch.randn(7, 48, 5) * 0.01; x[..., 4] = torch.rand(7, 48) / 48
    shifted = x.clone(); shifted[..., :4] += torch.randn(7, 1, 1) * 0.05      # same intraday shape, different day level
    with torch.no_grad():
        assert torch.allclose(enc(x), enc(shifted), atol=1e-5)
        assert not torch.allclose(IntraEnc(16, 1, cin=5, center=0).eval()(x), IntraEnc(16, 1, cin=5, center=0).eval()(shifted), atol=1e-3)


def test_epoch_days_uniform_is_a_permutation():
    days = np.arange(100, 400, 2)
    out = epoch_days(days, np.random.default_rng(0), 0)
    assert sorted(out) == sorted(days)


def test_epoch_days_recency_weights():
    days = np.arange(0, 1000)
    out = epoch_days(days, np.random.default_rng(0), 100)
    assert len(out) == len(days) and set(out) <= set(days)
    # with a 100-day half-life the newest 100 days carry ~half of the weight
    assert 0.45 < np.mean(out >= 900) < 0.55


def _synthetic(root, T=700, N=120):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2022-01-03', periods=T)
    codes = np.array([f'{i:06d}.SZ' for i in range(N)], dtype=object)
    root.mkdir()
    np.savez(root / 'axes.npz', dates=dates.to_numpy(), codes=codes)
    np.save(root / 'univ.npy', np.ones((T, N), bool))
    for k in ('y5', 'y10', 'y20'): np.save(root / f'{k}.npy', rng.normal(size=(T, N)).astype(np.float32))
    np.save(root / 'industry.npy', rng.integers(0, 5, (T, N)).astype(np.int16))
    np.save(root / 'log_circ_mv.npy', rng.normal(size=(T, N)).astype(np.float32))
    np.save(root / 'daily.npy', (rng.normal(size=(T, N, 10)) * 0.01).astype(np.float16))
    it = (rng.normal(size=(T, N, 48, 5)) * 0.01).astype(np.float16); it[..., 4] = 1 / 48
    np.save(root / 'intra.npy', it)
    np.save(root / 'market.npy', rng.normal(size=(T, 10)).astype(np.float32))
    np.save(root / 'extra.npy', rng.normal(size=(T, N, 6)).astype(np.float16))


OPTION_SETS = [
    ['--label', 'y5+y10+y20', '--intra_center', '1', '--half_life', '120'],
    ['--mkt', '1', '--head', 'glu'],
    ['--mkt', '2', '--rnn', 'lstm', '--no_intra', '1'],
    ['--extra', '1'],
    ['--extra', '1', '--mkt', '2', '--no_intra', '1'],
]


@pytest.mark.parametrize('opts', OPTION_SETS, ids=lambda o: ' '.join(o))
def test_tiny_run_with_new_options(tmp_path, opts):
    data = tmp_path / 'e2e'; _synthetic(data)
    cmd = [sys.executable, str(Path(train_mftg.__file__)), '--data', str(data), '--out', str(tmp_path / 'out'), '--name', 'tiny',
           '--blocks', '2024H2', '--seeds', '7', '--neutral', '1', '--refit', '1', '--epochs', '2',
           '--patience', '1', '--lookback', '20', '--days_intra', '3', '--d', '8', '--hidden', '8', '--layers', '1',
           '--stride', '10', '--device', 'cpu', '--amp', '0'] + opts
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    z = np.load(tmp_path / 'out' / 'tiny' / '2024H2_pred.npz')
    assert np.isfinite(z['pred']).all() and z['feature_date'].min() >= 20240701 and z['feature_date'].max() <= 20241231
    a = json.loads((tmp_path / 'out' / 'tiny' / 'args.json').read_text())
    for k, v in zip(opts[::2], opts[1::2]):
        got = a[k[2:]]; assert (float(got) == float(v)) if isinstance(got, (int, float)) else got == v


def test_market_gate_starts_open():
    torch.manual_seed(0)
    m = MFTG(d=8, h=8, layers=1, intra=False, mkt=2, nmk=10).eval()
    d = torch.randn(5, 20, 20) * 0.01; d2 = d.clone(); d2[..., 10:] = torch.randn(5, 20, 10)   # different market state
    with torch.no_grad():
        assert torch.allclose(m(d, None), m(d2, None), atol=1e-6)


def test_extra_inputs_reach_the_model():
    torch.manual_seed(0)
    m = MFTG(d=8, h=8, layers=1, intra=False, nx=6).eval()
    d = torch.randn(5, 20, 16) * 0.01; d2 = d.clone(); d2[..., 10:] += 1.0
    with torch.no_grad():
        assert not torch.allclose(m(d, None), m(d2, None), atol=1e-4)
