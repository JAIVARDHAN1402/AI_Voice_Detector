"""Publish the conformal study for the website and VoiceLock: compact results.json for web/research.html, the per-tier audit and strict
limits for VoiceLock (web/vl/conformal.json, read by both the Python service and the browser build) and the figures.

    python scripts/export_research_web.py
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
WEB = ROOT / "web" / "research"
METHODS = ("fixed_eer", "naive", "domain_robust")


def coverage(rep, summary=None):
    S = summary or rep["summary"]
    return {m: [{"alpha": a, "corpora": s["pct_corpora_meeting_alpha"], "meet": s["pct_generators_meeting_alpha"], "lo": s["ci95"][0],
                 "hi": s["ci95"][1], "mean_miss": s["mean_generator_miss"], "worst_miss": s["worst_generator_miss"],
                 "frr": s["genuine_rejected"], "units": s["units"], "n_corpora": s["corpora"]}
                for a in rep["alphas"] for s in [S[f"{m}@{a}"]]] for m in METHODS}


def main():
    mlp = json.loads((REPORTS / "conformal_mlp.json").read_text())
    lin_path = REPORTS / "conformal_linear.json"
    lin = json.loads(lin_path.read_text()) if lin_path.exists() else None
    dep = mlp["deployment_calibration"]
    data = {"delta": mlp["delta"], "alphas": mlp["alphas"], "domains": mlp["domains"], "coverage": coverage(mlp),
            "coverage_linear": coverage(lin) if lin else None,
            "other_scopes": {k: coverage(mlp, v) for k, v in mlp["other_scopes"].items()},
            "audit": mlp["audit_default_limits"], "in_domain_check": mlp["naive_in_domain_check"],
            "delta_sweep": mlp["delta_sweep_at_alpha_10pct"], "splice": mlp["splice"], "deployment": dep}
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / "results.json").write_text(json.dumps(data, indent=1))
    for f in (REPORTS / "figures").glob("conformal_*_mlp.png"):
        shutil.copy(f, WEB / f.name)
    cal = {"method": dep["method"], "delta": dep["delta"], "generators_used": dep["generators_used"], "clips_used": dep["clips_used"],
           "detector_threshold": dep["detector_threshold"], "scope": dep["scope"], "tiers": dep["tiers"], "audit": mlp["audit_default_limits"]}
    (ROOT / "web" / "vl" / "conformal.json").write_text(json.dumps(cal, indent=1))
    print("exported", WEB / "results.json", "and conformal.json for VoiceLock")


if __name__ == "__main__":
    main()
