"""Tests for the conformal thresholds (voicedetect.conformal) and the calibrated limits in VoiceLock."""
import numpy as np
import pytest

from voicedetect import conformal as C


def _units(rng, n_units, n_clips=60, spread=1.0):
    """AI-clip scores per generator: every generator has its own mean (the 'unseen generator' shift)."""
    return [rng.normal(rng.normal(2.0, spread), 1.0, n_clips) for _ in range(n_units)]


def test_naive_tau_meets_alpha_in_distribution():
    rng = np.random.default_rng(0)
    miss = []
    for _ in range(200):
        cal, new = rng.normal(2, 1, 500), rng.normal(2, 1, 2000)
        miss.append(C.miss_rate(new, C.naive_tau(cal, 0.05)))
    assert np.mean(miss) <= 0.05 + 0.005


def test_naive_tau_fails_for_unseen_generators_and_domain_robust_holds():
    rng = np.random.default_rng(1)
    alpha, delta = 0.05, 0.10
    ok_naive, ok_dr, trials = 0, 0, 0
    for _ in range(40):
        cal = _units(rng, 40)
        new = _units(rng, 25)
        t_naive = C.naive_tau(np.concatenate(cal), alpha)
        t_dr = C.domain_robust_tau(cal, alpha, delta)
        for u in new:
            trials += 1
            ok_naive += C.miss_rate(u, t_naive) <= alpha
            ok_dr += C.miss_rate(u, t_dr) <= alpha
    assert ok_naive / trials < 0.75            # the pooled guarantee does not transfer to a new generator
    assert ok_dr / trials >= 1 - delta - 0.03  # the generator-level guarantee does
    assert t_dr < t_naive                      # ...and it is paid for with a lower (stricter) threshold


def test_domain_robust_tau_is_monotone_in_alpha_and_delta():
    rng = np.random.default_rng(2)
    cal = _units(rng, 50)
    assert C.domain_robust_tau(cal, 0.01) <= C.domain_robust_tau(cal, 0.05) <= C.domain_robust_tau(cal, 0.20)
    assert C.domain_robust_tau(cal, 0.05, delta=0.02) <= C.domain_robust_tau(cal, 0.05, delta=0.20)


def test_small_generators_are_ignored_and_empty_input_raises():
    rng = np.random.default_rng(3)
    big = _units(rng, 20)
    tiny = [rng.normal(-50, 1, C.MIN_UNIT - 1)]          # would drag the threshold down if it were counted
    assert C.domain_robust_tau(big + tiny, 0.05) == C.domain_robust_tau(big, 0.05)
    with pytest.raises(ValueError):
        C.domain_robust_tau(tiny, 0.05)
    with pytest.raises(ValueError):
        C.naive_tau([], 0.05)


def test_tiny_alpha_flags_everything_instead_of_guessing():
    cal = np.linspace(-1, 3, 20)
    assert C.naive_tau(cal, 0.001) < cal.min()           # k = 0: no finite-sample guarantee possible -> flag all


def test_tau_to_p_ai_maps_the_detector_threshold_to_one_half():
    thr = 0.8202
    assert C.tau_to_p_ai(np.log(thr / (1 - thr)), thr) == pytest.approx(0.5)
    assert C.tau_to_p_ai(-3.0, thr) < 0.5 < C.tau_to_p_ai(3.0, thr)
