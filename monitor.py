"""
Monitoring: quality signals over time, drift against the reference set, alerts.

The thresholds below are PLACEHOLDERS. Choose yours and justify them in
REPORT.md: how many false alarms on normal inputs you accept, and how big a
shift has to be before it matters for your product.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REFERENCE_FILE = ROOT / "monitoring" / "reference_features.csv"

PSI_BINS = 5           # quantile bins of the reference; few, because samples are small
PSI_ALERT = 0.25       # PLACEHOLDER: alert when a feature's PSI exceeds this...
NOISE_MULTIPLE = 2     # ...and this many times the PSI expected by chance (psi_noise)
WINDOW = 100           # the drift check looks at the last WINDOW requests...
INPUT_SIGNALS = {"mean_read_confidence","reading_count","unassigned_answers","ink_density","line_count"}
MIN_WINDOW = 20        # ...and stays silent with fewer than this
THUMBS_DOWN_ALERT = 0.4  # PLACEHOLDER: alert when this share of rated requests is 👎


def _bin(values: np.ndarray, reference: np.ndarray, bins: int) -> np.ndarray:
    """Bin index of each value, with bins defined by the reference.

    A feature with few distinct values in the reference (counts, labels coded as
    numbers) gets one bin per value plus one for anything else; a continuous one
    gets quantile bins, so each holds about the same share of the reference.
    """
    distinct = np.unique(reference)
    if len(distinct) <= bins:
        index = np.searchsorted(distinct, values)
        found = (index < len(distinct)) & (distinct[np.minimum(index, len(distinct) - 1)] == values)
        return np.where(found, index, len(distinct))
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, values, side="right")


def psi(reference, current, bins: int = PSI_BINS, *, pseudocount: float = 0) -> float:
    """Population stability index of `current` against `reference`.

    0 means the same distribution; the usual rule of thumb is < 0.1 stable,
    0.1-0.25 moderate, > 0.25 a large shift. With small samples, though, PSI is
    large by chance alone: see `psi_noise`.
    """
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    ref_bins, cur_bins = _bin(reference, reference, bins), _bin(current, reference, bins)
    n_bins = max(ref_bins.max(), cur_bins.max()) + 1
    if not np.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError("Use a finite nonnegative bin pseudocount.")
    if pseudocount:
        distinct = np.unique(reference)
        n_bins = len(distinct) + 1 if len(distinct) <= bins else len(np.unique(
            np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))) + 1
    def proportions(indices):
        counts = np.bincount(indices, minlength=n_bins) + pseudocount
        shares = counts / counts.sum()
        return shares if pseudocount else np.clip(shares, 1e-4, None)
    expected, actual = proportions(ref_bins), proportions(cur_bins)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_noise(n_reference: int, n_current: int, bins: int = PSI_BINS) -> float:
    """PSI you would expect by chance, with no drift at all, for these sample sizes.

    About (bins - 1) * (1/n_reference + 1/n_current): with 40 reference inputs and
    20 live ones that is already 0.3. An alert below this is a coin toss.
    """
    return (bins - 1) * (1 / n_reference + 1 / n_current)


def load_reference() -> pd.DataFrame | None:
    """Features of the reference inputs (`main.py reference` writes them)."""
    return pd.read_csv(REFERENCE_FILE) if REFERENCE_FILE.is_file() else None


def drift(reference: pd.DataFrame, current: pd.DataFrame, min_current: int=2, *, pseudocount: float=0) -> pd.DataFrame:
    """PSI per feature, and whether it alerts. Both frames have one column per feature (no `f_` prefix).

    The rule: alert when PSI exceeds PSI_ALERT *and* NOISE_MULTIPLE times what
    chance alone would give with these sample sizes.
    """
    rows = []
    for feature in reference.columns:
        if feature in current.columns and feature in INPUT_SIGNALS:
            ref=reference[feature].dropna()
            cur=current[feature].dropna()
            if len(ref)<30 or len(cur)<min_current:
                continue
            value = psi(ref, cur, pseudocount=pseudocount)
            noise = psi_noise(len(ref),len(cur))
            threshold = max(PSI_ALERT, NOISE_MULTIPLE * noise)
            rows.append({"feature": feature, "psi": round(value, 3), "threshold": round(threshold, 3), "alert": bool(value > threshold)})
    return pd.DataFrame(rows, columns=["feature", "psi", "threshold", "alert"])


def live_features(logs: pd.DataFrame) -> pd.DataFrame:
    """The features of the last WINDOW logged requests, without the `f_` prefix."""
    columns = [c for c in logs.columns if c.startswith("f_")]
    return logs[columns].tail(WINDOW).rename(columns=lambda c: c[2:])


def quality(logs: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Quality signals per period, in long format (time, signal, value) for a line plot."""
    if logs.empty:
        return pd.DataFrame(columns=["time", "signal", "value"])
    df = logs.set_index("timestamp")
    rated = df["feedback"].dropna().astype(float)
    signals = pd.DataFrame({
        "share sent to review": df["review"].astype(float).resample(freq).mean(),
        "share 👎 (of rated)": (rated < 0).astype(float).resample(freq).mean(),
    })
    if "f_confidence" in df:
        signals["mean confidence"] = df["f_confidence"].resample(freq).mean()
    long = signals.reset_index(names="time").melt("time", var_name="signal", value_name="value")
    return long.dropna()


def summary(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame({"metric": ["requests"], "value": [0]})
    rated = logs["feedback"].dropna().astype(float)
    def known_price(outputs):
        component = outputs.get("total_usage") if isinstance(outputs, dict) else None
        if component is None and isinstance(outputs, dict):
            grader = outputs.get("component_b")
            if isinstance(grader, dict) and grader.get("input_type") in {"ocr_export", "pdf_awaiting_reader"}:
                component = grader
        return isinstance(component, dict) and component.get("cost_known") is True
    known = logs["outputs"].map(known_price)
    priced = logs.loc[known,"cost_eur"]
    rows = {
        "requests": len(logs),
        "sent to review": f"{logs['review'].mean():.0%}",
        "rated 👍/👎": len(rated),
        "share 👎 (of rated)": f"{(rated < 0).mean():.0%}" if len(rated) else "-",
        "median latency (s)": round(float(logs["latency_s"].median()), 2),
        "requests with known cost": len(priced),
        "mean known request cost (€)": round(float(priced.mean()), 5) if len(priced) else "-",
        "total of known costs (€)": round(float(priced.sum()), 4) if len(priced) else "-",
    }
    return pd.DataFrame({"metric": list(rows), "value": [str(v) for v in rows.values()]})


def alerts(logs: pd.DataFrame, reference: pd.DataFrame | None) -> list[str]:
    """The alert rules. Each fired alert is one message."""
    fired = []
    window = logs.tail(WINDOW)
    if reference is not None and len(window) >= MIN_WINDOW:
        report = drift(reference, live_features(window),MIN_WINDOW)
        for row in report[report["alert"]].itertuples():
            fired.append(f"Drift in `{row.feature}`: PSI {row.psi} > {row.threshold} over the last {len(window)} requests.")
    rated = window["feedback"].dropna().astype(float) if not window.empty else pd.Series(dtype=float)
    if len(rated) >= MIN_WINDOW and (rated < 0).mean() > THUMBS_DOWN_ALERT:
        fired.append(f"{(rated < 0).mean():.0%} of the last {len(rated)} rated requests are 👎 (> {THUMBS_DOWN_ALERT:.0%}).")
    return fired

def drift_status(logs, reference):
    if reference is None or len(reference)<30:
        return f"Input drift not assessed: {0 if reference is None else len(reference)}/30 reference observations."
    if len(logs)<MIN_WINDOW:
        return f"Input drift not assessed: {len(logs)}/{MIN_WINDOW} recent requests."
    report=drift(reference,live_features(logs),MIN_WINDOW)
    if report.empty:
        return "Input drift not assessed: insufficient comparable input observations."
    return "Input drift assessed on "+", ".join(report.feature)+(" (alert)." if report.alert.any() else " (no alert).")
