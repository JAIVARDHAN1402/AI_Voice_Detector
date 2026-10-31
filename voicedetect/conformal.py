"""Conformal thresholds for the AI-voice score, with a guarantee that survives an unseen generator.

Score convention: a clip's score is the mean window logit of the detector; larger = more AI-like. A clip is flagged as AI when
score > tau. The error that matters for authorisation is a *miss*: an AI clip with score <= tau is accepted as real.

naive_tau          split conformal over pooled calibration clips. P(miss) <= alpha for a new clip drawn from the SAME
                   distribution as the calibration clips (exchangeability). A new TTS system breaks that assumption.
domain_robust_tau  the exchangeable unit is the GENERATOR, not the clip: compute each calibration generator's own
                   alpha-quantile threshold, then take the delta-quantile of those thresholds. For a new generator
                   exchangeable with the calibration generators, P(its miss rate <= alpha) >= 1 - delta (approximately: each
                   generator's quantile is estimated from finitely many clips).
"""
import numpy as np

MIN_UNIT = 8      # generators with fewer clips are ignored (their own quantile is too noisy)


def naive_tau(fake_scores, alpha: float) -> float:
    f = np.sort(np.asarray(fake_scores, float))
    if not len(f):
        raise ValueError("no calibration scores")
    k = int(np.floor(alpha * (len(f) + 1)))
    return float(f[k - 1] if k >= 1 else f[0] - 1.0)


def domain_robust_tau(units, alpha: float, delta: float = 0.10, min_unit: int = MIN_UNIT) -> float:
    """units: list of arrays, the AI-clip scores of each calibration generator."""
    t = np.sort([np.quantile(np.asarray(u, float), alpha, method="lower") for u in units if len(u) >= min_unit])
    if not len(t):
        raise ValueError("no calibration generator has enough clips")
    k = int(np.floor(delta * (len(t) + 1)))
    return float(t[k - 1] if k >= 1 else t[0] - 1.0)


def miss_rate(fake_scores, tau: float) -> float:
    return float((np.asarray(fake_scores) <= tau).mean())


def tau_to_p_ai(tau: float, detector_threshold: float) -> float:
    """Raw mean-logit threshold -> the calibrated P(AI) the detector reports (sigmoid(mean logit - logit(threshold)))."""
    lt = np.log(detector_threshold / (1 - detector_threshold))
    return float(1 / (1 + np.exp(-(tau - lt))))
