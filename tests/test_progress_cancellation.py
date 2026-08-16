"""Progress reporting + cooperative cancellation hooks (GUI/back-end API)."""

import numpy as np
import pytest

from hifi_anova import hifi_anova, HiFiCancelled
from hifi_anova.progress import make_event, stage_fraction


def _toy_data(n=400, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.5 * (X[:, 1] - 0.5) + 0.1 * rng.standard_normal(n)
    return X, y


def test_progress_events_fire_in_order():
    X, y = _toy_data()
    events = []
    hifi_anova(X, y, K1=4, K2=2, mode='second', verbose=False,
               progress=events.append)

    kinds = [e['event'] for e in events]
    assert kinds[0] == 'fit_start'
    assert kinds[-1] == 'done'
    # Stage A and B ran; each brackets a start/end.
    stages_started = [e['stage'] for e in events if e['event'] == 'stage_start']
    assert 'A' in stages_started and 'B' in stages_started
    assert any(e['event'] == 'phase' and e['stage'] == 'analysis' for e in events)
    # Every event carries the schema keys and a monotone-ish fraction in [0,1].
    for e in events:
        assert set(('event', 'stage', 'fraction')) <= set(e)
        assert 0.0 <= e['fraction'] <= 1.0
    assert events[-1]['fraction'] == 1.0


def test_progress_emits_stage_d_iterations():
    X, y = _toy_data(n=600)
    events = []
    hifi_anova(X, y, K1=4, K2=2, Kh=3, heteroscedastic=True, verbose=False,
               progress=events.append)
    assert any(e['event'] == 'stage_start' and e['stage'] == 'D' for e in events)
    # Fine-grained Stage-D outer-iteration progress is emitted.
    d_prog = [e for e in events if e['event'] == 'stage_progress' and e['stage'] == 'D']
    assert d_prog, "expected per-outer-iteration Stage-D progress events"
    assert 'outer' in d_prog[0]['metrics']


def test_should_stop_cancels_fit():
    X, y = _toy_data()
    seen = []

    def stop():
        seen.append(1)
        return True   # cancel at the very first checkpoint

    with pytest.raises(HiFiCancelled):
        hifi_anova(X, y, K1=4, K2=2, mode='second', verbose=False,
                   should_stop=stop)
    assert seen, "should_stop was never polled"


def test_should_stop_false_completes_normally():
    X, y = _toy_data()
    res = hifi_anova(X, y, K1=4, K2=2, mode='second', verbose=False,
                     should_stop=lambda: False)
    assert res.model is not None


def test_no_hooks_is_unchanged_default():
    # The default path (no progress/should_stop) must behave exactly as before.
    X, y = _toy_data()
    res = hifi_anova(X, y, K1=4, K2=2, mode='second', verbose=False)
    assert res.model is not None


def test_make_event_schema():
    ev = make_event('stage_end', stage='B', metrics={'rmse_val': 0.1})
    assert ev['event'] == 'stage_end' and ev['stage'] == 'B'
    assert ev['fraction'] == stage_fraction('B')
    assert ev['metrics']['rmse_val'] == 0.1
    # Unknown stage → 0.0 fraction; explicit fraction overrides.
    assert make_event('x', stage='zzz')['fraction'] == 0.0
    assert make_event('done', fraction=1.0)['fraction'] == 1.0
