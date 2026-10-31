"""Stage 2 of the conformal study: how often is an AI voice from an *unseen* generator accepted as real, and can that rate be bounded?

Question
    VoiceLock needs "an AI voice is accepted as real at most alpha of the time". Split-conformal calibration gives that for a new
    clip drawn from the calibration distribution. A new TTS system / corpus is not from that distribution, so we measure how the
    guarantee degrades and compare three ways of choosing the threshold (all on the same clip score = mean window logit):

    fixed_eer     one threshold at the pooled equal-error point (what the app used before; ignores alpha)
    naive         split conformal over pooled calibration clips
    domain_robust per-generator alpha-quantile thresholds, then the delta-quantile ACROSS generators (generator = exchangeable unit)

Protocol: nested leave-one-domain-out (scripts/conformal_lodo.py). Calibration scores of a domain come from detectors that never
saw it or the held-out domain; test scores come from a detector that never saw the held-out domain.

Three scopes (capture conditions) are reported because they behave very differently:
    direct   clean + augmented (codec / noise / reverb / band-limit): audio injected or uploaded, or a genuine user's microphone. PRIMARY.
    all      everything, including simulated loudspeaker -> room -> microphone replay.
    replay   only the replay-simulated clips (the detector is close to chance there).

Two validity levels are reported: per GENERATOR (fraction of held-out generators with miss <= alpha) and per CORPUS (fraction of
held-out corpora whose pooled miss <= alpha). Generators cluster by corpus, so the corpus level is the honest unit of exchangeability.

    python scripts/conformal_study.py [--head mlp|linear]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
from sklearn.metrics import roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voicedetect.conformal import MIN_UNIT, domain_robust_tau as tau_domain_robust, naive_tau as tau_naive  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "conformal"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"

ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.40]
TIER_ALPHA = {"low": 0.30, "medium": 0.20, "high": 0.10, "critical": 0.05}   # targets of the optional strict mode
HAND_LIMITS = {"low": 0.50, "medium": 0.45, "high": 0.35, "critical": 0.25}   # the default P(AI) limits in voicelock/config.py
SCOPES = {"direct": ("clean", "augmented"), "all": None, "replay": ("replay",)}
DELTA = 0.10          # a new generator meets its alpha with probability >= 1 - DELTA (approximately; see paper)
NMIN = MIN_UNIT       # smallest generator used for calibration
EVAL_MIN = 20         # smallest held-out generator whose miss rate is scored (fewer clips make 'miss <= alpha' too noisy)
RNG = np.random.default_rng(7)
BLUE, GOLD, GREEN, RED, GREY = "#3b5bdb", "#d9a21b", "#2f855a", "#c53030", "#6b7280"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
METHODS = ("fixed_eer", "naive", "domain_robust")


# ---------------------------------------------------------------- data
def load(head):
    with open(OUT / f"lodo_scores_{head}.pkl", "rb") as f:
        return pickle.load(f)


def clip_table(meta, z, conds=None):
    """window logits -> one row per (clip, condition): mean logit, label, generator, condition (optionally only some conditions).
    The detector saturates (probabilities are clipped at 1e-4, i.e. logits at +-9.21), which creates exact ties; conformal needs
    continuous scores, so ties are broken with a tiny fixed noise (randomised conformal; far below any decision margin)."""
    z = np.asarray(z, np.float64) + np.random.default_rng(len(z)).uniform(0, 1e-3, len(z))
    key = meta["clip"].astype(np.int64) * 4 + meta["aug"].astype(np.int64) * 2 + meta["replay"].astype(np.int64)
    u, inv = np.unique(key, return_inverse=True)
    s = np.bincount(inv, z) / np.bincount(inv)
    y = np.zeros(len(u), int); y[inv] = meta["y"]
    sysm = np.empty(len(u), object); sysm[inv] = meta["system"]
    cond = np.empty(len(u), object); cond[inv] = np.where(meta["replay"], "replay", np.where(meta["aug"], "augmented", "clean"))
    t = {"s": s, "y": y, "sys": sysm, "cond": cond}
    if conds is not None:
        m = np.isin(cond, list(conds))
        t = {k: v[m] for k, v in t.items()}
    return t


def fake_units(ct):
    return [ct["s"][(ct["y"] == 1) & (ct["sys"] == g)] for g in sorted(set(ct["sys"][ct["y"] == 1]))
            if ((ct["y"] == 1) & (ct["sys"] == g)).sum() >= NMIN]


def tau_eer(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.nanargmin(np.abs((1 - tpr) - fpr)))])


def _tables(D, H, conds):
    meta, sc = H["meta"], H["scores"]
    test = clip_table(meta[D], sc[(D,)][D], conds)
    cal = {r: clip_table(meta[r], sc[tuple(sorted((D, r)))][r], conds) for r in H["domains"] if r != D}
    return test, cal


def _test_units(test):
    out = [(g, test["s"][(test["y"] == 1) & (test["sys"] == g)]) for g in sorted(set(test["sys"][test["y"] == 1]))]
    return [(g, u) for g, u in out if len(u) >= EVAL_MIN]


# ---------------------------------------------------------------- the leave-one-domain-out evaluation
def evaluate(D, H, alphas=ALPHAS, delta=DELTA, conds=None):
    test, cal = _tables(D, H, conds)
    cal_fake = np.concatenate([c["s"][c["y"] == 1] for c in cal.values()])
    cal_units = [u for c in cal.values() for u in fake_units(c)]
    cal_y = np.concatenate([c["y"] for c in cal.values()]); cal_s = np.concatenate([c["s"] for c in cal.values()])
    real = test["s"][test["y"] == 0]
    test_units = _test_units(test)
    taus = {"fixed_eer": tau_eer(cal_y, cal_s)}
    for a in alphas:
        taus[("naive", a)] = tau_naive(cal_fake, a)
        taus[("domain_robust", a)] = tau_domain_robust(cal_units, a, delta)
    rows = []
    for method in METHODS:
        for a in alphas:
            tau = taus["fixed_eer"] if method == "fixed_eer" else taus[(method, a)]
            rows.append({"domain": D, "method": method, "alpha": a, "tau": tau,
                         "unit_miss": [float((u <= tau).mean()) for _, u in test_units], "unit_names": [g for g, _ in test_units],
                         "pooled_miss": float((np.concatenate([u for _, u in test_units]) <= tau).mean()) if test_units else float("nan"),
                         "frr": float((real > tau).mean()) if len(real) >= NMIN else None, "n_real": int(len(real)),
                         "n_cal_units": len(cal_units)})
    # sanity check: ordinary in-domain calibration (half of D's AI clips calibrate, the other half test) must meet alpha
    fk = test["s"][test["y"] == 1]; perm = RNG.permutation(len(fk)); h = len(fk) // 2
    for a in alphas:
        t = tau_naive(fk[perm[:h]], a)
        rows.append({"domain": D, "method": "naive_in_domain", "alpha": a, "tau": t, "unit_miss": [], "unit_names": [],
                     "pooled_miss": float((fk[perm[h:]] <= t).mean()), "frr": float((real > t).mean()) if len(real) >= NMIN else None,
                     "n_real": int(len(real)), "n_cal_units": 0})
    return rows


def boot_ci(vals, n=2000):
    vals = np.asarray(vals, float)
    b = vals[RNG.choice(len(vals), (n, len(vals)))].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summarise(rows, alphas=ALPHAS):
    out = {}
    for method in METHODS:
        for a in alphas:
            rs = [r for r in rows if r["method"] == method and r["alpha"] == a]
            miss = np.concatenate([r["unit_miss"] for r in rs])
            ok = (miss <= a).astype(float)
            pooled = np.array([r["pooled_miss"] for r in rs if not np.isnan(r["pooled_miss"])])
            frrs = [r["frr"] for r in rs if r["frr"] is not None]
            out[f"{method}@{a}"] = {
                "units": int(len(miss)), "pct_generators_meeting_alpha": float(ok.mean()), "ci95": boot_ci(ok),
                "pct_corpora_meeting_alpha": float((pooled <= a).mean()), "corpora": int(len(pooled)),
                "mean_generator_miss": float(miss.mean()), "median_generator_miss": float(np.median(miss)),
                "worst_generator_miss": float(miss.max()), "genuine_rejected": float(np.mean(frrs)),
                "per_domain": {r["domain"]: {"frr": r["frr"], "pooled_miss": r["pooled_miss"], "tau": r["tau"]} for r in rs}}
    for a in alphas:
        rs = [r for r in rows if r["method"] == "naive_in_domain" and r["alpha"] == a]
        out[f"naive_in_domain@{a}"] = {"mean_miss": float(np.mean([r["pooled_miss"] for r in rs]))}
    return out


# ---------------------------------------------------------------- audit of the default tier limits
def audit_limits(H, conds, thr, limits=HAND_LIMITS):
    """What do the limits VoiceLock ships with actually do on generators/corpora the detector has never seen?
    P(AI) limit p  ->  raw mean-logit threshold  z = logit(p) + logit(thr)  (the detector reports P(AI) = sigmoid(z - logit(thr)))."""
    lg = lambda q: float(np.log(q / (1 - q)))  # noqa: E731
    res = {}
    tabs = {D: _tables(D, H, conds)[0] for D in H["domains"]}
    for tier, p in limits.items():
        z = lg(p) + lg(thr)
        miss, pooled, frr = [], [], []
        for D, t in tabs.items():
            tu = _test_units(t)
            miss += [float((u <= z).mean()) for _, u in tu]
            if tu:
                pooled.append(float((np.concatenate([u for _, u in tu]) <= z).mean()))
            real = t["s"][t["y"] == 0]
            if len(real) >= NMIN:
                frr.append(float((real > z).mean()))
        miss = np.array(miss)
        res[tier] = {"p_ai_limit": p, "tau_logit": z, "generators": int(len(miss)), "mean_generator_miss": float(miss.mean()),
                     "median_generator_miss": float(np.median(miss)), "pct_generators_miss_over_10pct": float((miss > 0.10).mean()),
                     "pct_generators_miss_over_25pct": float((miss > 0.25).mean()), "mean_corpus_miss": float(np.mean(pooled)),
                     "genuine_rejected": float(np.mean(frr))}
    return res


# ---------------------------------------------------------------- splice study (window level)
def splice_study(H, M=4, n=20000, fpr=0.05):
    """Pseudo-clips of M windows, k of them AI (a cloned challenge pasted into a real recording); window scores are the
    out-of-domain LODO scores of direct-capture audio. Window boundaries/co-articulation at the seam are NOT modelled.
    Ties (the detector saturates) are broken with tiny noise, i.e. a randomised test."""
    meta, sc = H["meta"], H["scores"]
    real_w, fake_w = [], []
    for d in H["domains"]:
        z, m = sc[(d,)][d], meta[d]
        direct = ~m["replay"]
        real_w.append(z[(m["y"] == 0) & direct]); fake_w.append(z[(m["y"] == 1) & direct])
    real_w, fake_w = np.concatenate(real_w), np.concatenate(fake_w)
    jit = lambda a: a + RNG.uniform(0, 1e-3, a.shape)  # noqa: E731
    agg = {"mean (deployed)": lambda Z: Z.mean(1), "median": lambda Z: np.median(Z, 1), "max": lambda Z: Z.max(1),
           "mean of top-2": lambda Z: np.sort(Z, 1)[:, -2:].mean(1)}

    def make(k):
        if k == 0:
            return jit(RNG.choice(real_w, (n, M)))
        if k == M:
            return jit(RNG.choice(fake_w, (n, M)))
        return jit(np.concatenate([RNG.choice(real_w, (n, M - k)), RNG.choice(fake_w, (n, k))], axis=1))
    base = {k: make(k) for k in range(M + 1)}
    res = {}
    for name, f in agg.items():
        thr = np.quantile(f(base[0]), 1 - fpr)
        res[name] = {f"{k}/{M} windows AI": float((f(base[k]) > thr).mean()) for k in range(1, M + 1)}
    return {"M": M, "false_alarm_on_pure_real": fpr, "real_windows": int(len(real_w)), "ai_windows": int(len(fake_w)), "detection": res}


# ---------------------------------------------------------------- deployment calibration (used by VoiceLock strict mode)
def deployment_calibration(H, conds=SCOPES["direct"]):
    """Every domain is 'unseen' in its own LODO fold, so the pooled single-fold scores are the calibration set for the
    detector that is actually shipped (trained on all domains)."""
    meta, sc = H["meta"], H["scores"]
    tabs = {d: clip_table(meta[d], sc[(d,)][d], conds) for d in H["domains"]}
    units = [u for t in tabs.values() for u in fake_units(t)]
    allfake = np.concatenate([t["s"][t["y"] == 1] for t in tabs.values()])
    allreal = np.concatenate([t["s"][t["y"] == 0] for t in tabs.values()])
    thr = float(joblib.load(ROOT / "models" / "detector.joblib")["threshold"]); lthr = float(np.log(thr / (1 - thr)))
    sig = lambda z: float(1 / (1 + np.exp(-(z - lthr))))  # noqa: E731
    tiers = {}
    for tier, a in TIER_ALPHA.items():
        tau, tn = tau_domain_robust(units, a), tau_naive(allfake, a)
        tiers[tier] = {"alpha": a, "tau_logit": tau, "p_ai_limit": sig(tau), "genuine_rejected_pooled": float((allreal > tau).mean()),
                       "tau_naive_logit": tn, "p_ai_limit_naive": sig(tn)}
    return {"method": "domain-robust conformal (generator = exchangeable unit), direct-capture audio", "delta": DELTA,
            "generators_used": len(units), "clips_used": int(len(allfake) + len(allreal)), "detector_threshold": thr, "tiers": tiers,
            "scope": "direct"}


# ---------------------------------------------------------------- figures
def figures(S, rows, splice, head):
    FIG.mkdir(parents=True, exist_ok=True)
    names = {"fixed_eer": ("fixed threshold (before)", GREY), "naive": ("naive conformal", GOLD), "domain_robust": ("domain-robust (ours)", BLUE)}
    x = np.arange(len(ALPHAS))
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for i, m in enumerate(METHODS):
        off = (i - 1) * 0.07
        ax[0].plot(x + off, [100 * S[f"{m}@{a}"]["pct_corpora_meeting_alpha"] for a in ALPHAS], marker="o", ms=4, lw=1.3, color=names[m][1], label=names[m][0])
        v = [100 * S[f"{m}@{a}"]["pct_generators_meeting_alpha"] for a in ALPHAS]
        lo = [100 * S[f"{m}@{a}"]["ci95"][0] for a in ALPHAS]; hi = [100 * S[f"{m}@{a}"]["ci95"][1] for a in ALPHAS]
        ax[1].errorbar(x + off, v, yerr=[np.array(v) - lo, np.array(hi) - v], marker="o", ms=4, lw=1.3, color=names[m][1], capsize=2)
        ax[2].plot(x, [100 * S[f"{m}@{a}"]["genuine_rejected"] for a in ALPHAS], marker="o", ms=4, lw=1.3, color=names[m][1])
    for a_ in ax[:2]:
        a_.axhline(100 * (1 - DELTA), color="k", ls=":", lw=0.8); a_.set_ylim(0, 105)
    ax[0].set_ylabel("held-out corpora meeting alpha (%)"); ax[1].set_ylabel("held-out generators meeting alpha (%)"); ax[2].set_ylabel("genuine speech rejected (%)")
    ax[0].legend(frameon=False, fontsize=6.3, loc="lower right")
    for a_ in ax:
        a_.set_xticks(x); a_.set_xticklabels([f"{int(a * 100)} %" for a in ALPHAS]); a_.set_xlabel("allowed AI-accepted rate (alpha)")
    fig.tight_layout(); fig.savefig(FIG / f"conformal_coverage_{head}.png", dpi=200, facecolor="white"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    a = 0.10
    data = [np.concatenate([r["unit_miss"] for r in rows if r["method"] == m and r["alpha"] == a]) * 100 for m in METHODS]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, flierprops={"markersize": 2})
    for p, c in zip(bp["boxes"], (GREY, GOLD, BLUE)): p.set_facecolor(c); p.set_alpha(0.55)
    ax.axhline(a * 100, color="k", ls=":", lw=0.8); ax.text(3.45, a * 100 + 2, "alpha = 10 %", fontsize=6.5, ha="right")
    ax.set_xticklabels(["fixed", "naive", "domain-robust"]); ax.set_ylabel("AI accepted as real, per unseen generator (%)")
    fig.tight_layout(); fig.savefig(FIG / f"conformal_generators_{head}.png", dpi=200, facecolor="white"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for (name, vals), c in zip(splice["detection"].items(), (GREY, BLUE, GREEN, GOLD)):
        ax.plot(range(1, len(vals) + 1), [100 * v for v in vals.values()], marker="o", ms=4, lw=1.3, color=c, label=name)
    M = splice["M"]; ax.set_xticks(range(1, M + 1)); ax.set_xticklabels([f"{k}/{M}" for k in range(1, M + 1)])
    ax.set_xlabel("share of the recording that is AI (spliced)"); ax.set_ylabel(f"detected at {int(splice['false_alarm_on_pure_real'] * 100)} % false alarms (%)")
    ax.legend(frameon=False, fontsize=6.3); fig.tight_layout(); fig.savefig(FIG / f"conformal_splice_{head}.png", dpi=200, facecolor="white"); plt.close(fig)


def build_report(H):
    thr = float(joblib.load(ROOT / "models" / "detector.joblib")["threshold"])
    scopes = {}
    rows_by_scope = {}
    for name, conds in SCOPES.items():
        rows = [r for D in H["domains"] for r in evaluate(D, H, conds=conds)]
        rows_by_scope[name] = rows
        scopes[name] = summarise(rows)
    delta_rows = {}                                            # robustness over delta (direct scope)
    for dl in (0.05, 0.10, 0.20):
        rr = [r for D in H["domains"] for r in evaluate(D, H, alphas=[0.10], delta=dl, conds=SCOPES["direct"]) if r["method"] == "domain_robust"]
        miss = np.concatenate([r["unit_miss"] for r in rr]); pooled = np.array([r["pooled_miss"] for r in rr])
        delta_rows[str(dl)] = {"pct_generators_meeting_alpha_10pct": float((miss <= 0.10).mean()),
                               "pct_corpora_meeting_alpha_10pct": float((pooled <= 0.10).mean()),
                               "genuine_rejected": float(np.mean([r["frr"] for r in rr if r["frr"] is not None]))}
    S = scopes["direct"]
    report = {"head": H["head"], "delta": DELTA, "alphas": ALPHAS, "nmin": NMIN, "eval_min": EVAL_MIN, "primary_scope": "direct",
              "scopes_definition": {k: (list(v) if v else "all conditions") for k, v in SCOPES.items()},
              "summary": S, "other_scopes": {k: v for k, v in scopes.items() if k != "direct"},
              "audit_default_limits": audit_limits(H, SCOPES["direct"], thr), "splice": splice_study(H),
              "deployment_calibration": deployment_calibration(H), "delta_sweep_at_alpha_10pct": delta_rows, "domains": H["domains"],
              "naive_in_domain_check": {f"{al}": S[f"naive_in_domain@{al}"]["mean_miss"] for al in ALPHAS}}
    return report, rows_by_scope["direct"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--head", default="mlp", choices=["mlp", "linear"]); a = ap.parse_args()
    H = load(a.head)
    report, rows = build_report(H)
    S = report["summary"]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"conformal_{a.head}.json").write_text(json.dumps(report, indent=2))
    figures(S, rows, report["splice"], a.head)
    print(f"== head={a.head}  scope=direct (clean + augmented)")
    for m in METHODS:
        for al in ALPHAS:
            s = S[f"{m}@{al}"]
            print(f"{m:14s} a={al:<5} corpora meet={100 * s['pct_corpora_meeting_alpha']:4.0f}%  generators({s['units']}) meet={100 * s['pct_generators_meeting_alpha']:5.1f}% "
                  f"[{100 * s['ci95'][0]:.0f},{100 * s['ci95'][1]:.0f}] meanmiss={100 * s['mean_generator_miss']:5.1f}% "
                  f"worst={100 * s['worst_generator_miss']:5.1f}% genuineRejected={100 * s['genuine_rejected']:5.1f}%")
    print("in-domain sanity (should be ~alpha):", {k: round(v, 3) for k, v in report["naive_in_domain_check"].items()})
    print("delta sweep:", report["delta_sweep_at_alpha_10pct"])
    print("audit of default limits:", json.dumps(report["audit_default_limits"], indent=1))
    print("strict tiers:", json.dumps(report["deployment_calibration"]["tiers"], indent=1))
    print("splice:", json.dumps(report["splice"]["detection"], indent=1))
    for sc_name, sm in report["other_scopes"].items():
        for m in ("fixed_eer", "naive", "domain_robust"):
            s = sm[f"{m}@0.1"]
            print(f"[{sc_name}] {m:14s} a=0.1 corpora meet={100 * s['pct_corpora_meeting_alpha']:4.0f}% generators meet={100 * s['pct_generators_meeting_alpha']:5.1f}% genuineRejected={100 * s['genuine_rejected']:5.1f}%")


if __name__ == "__main__":
    main()
