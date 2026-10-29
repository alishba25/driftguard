"""Turn check results into a short list of alerts a person should look at."""
from driftguard import config


def make_alerts(batch: dict, quality_results: list, drift_results: list) -> list[dict]:
    alert_list = []

    # ---- Quality alerts ----
    if batch["status"] == "failed":
        reasons = "; ".join(r.message for r in quality_results if r.severity == "critical")
        alert_list.append({"category": "quality", "severity": "critical",
                           "message": f"Batch unusable: {reasons}"})
    elif batch["row_count"] and batch["rejected_count"] / batch["row_count"] > config.QUALITY_ALERT_RATE:
        rate = batch["rejected_count"] / batch["row_count"]
        alert_list.append({"category": "quality", "severity": "warning",
                           "message": f"{batch['rejected_count']} rows rejected ({rate:.1%})"})

    # ---- Drift alert: one per batch, naming the drifted features ----
    drifted = [d.feature for d in drift_results if d.drift_detected]
    if drifted:
        alert_list.append({"category": "drift", "severity": "warning",
                           "message": f"Distribution changed in: {', '.join(drifted)}"})
    return alert_list