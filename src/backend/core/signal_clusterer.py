"""
signal_clusterer.py — Density-based clustering of FAERS adverse-event reports.

Purpose
-------
Group individual adverse-event reports into coherent clusters so that
patterns across many reports can be surfaced at a glance.  Each cluster
is described in plain English: which drug dominates, which events dominate,
how severe the outcomes are, and how "pure" (single-drug) the cluster is.

Algorithm choice: DBSCAN
------------------------
We use DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
from scikit-learn rather than KMeans for the following reasons:

1.  Unknown number of clusters.
    KMeans requires k to be specified in advance.  Pharmacovigilance data
    has no natural k — the number of meaningful groups depends entirely on
    which drugs and events appear in a given dataset.  DBSCAN discovers k
    from the data density.

2.  Clusters of varying size.
    Warfarin may generate hundreds of reports while a rare drug generates
    three.  KMeans biases toward equal-size clusters.  DBSCAN has no such
    assumption.

3.  Noise handling.
    Isolated reports (a single drug–event pair seen once) do not belong to
    any meaningful cluster.  KMeans forces every point into a cluster.
    DBSCAN labels outliers as noise (cluster_id = -1) and excludes them
    from cluster statistics.

4.  No spherical-cluster assumption.
    KMeans minimises inertia, which implicitly assumes convex, roughly
    spherical clusters in Euclidean space.  Drug–event feature vectors are
    sparse and high-dimensional; density-based clustering is more appropriate.

Feature representation
----------------------
Each report is encoded as a binary (0/1) vector via one-hot encoding of:
    - drug_name       (which drug was involved)
    - adverse_event   (which MedDRA preferred term was reported)
    - outcome_code    (DE / HO / LT / DS / CA / RI / OT)

Age, sex, and report_date are excluded:
    - Age and sex are frequently missing in FAERS
    - Report date is continuous and would dominate cosine distance
    - The clustering question is "what happened", not "who it happened to"

Distance metric: cosine
-----------------------
Binary feature vectors are sparse.  Cosine distance measures the *angle*
between vectors rather than their Euclidean distance, which correctly
treats two reports as similar when they share drug/event/outcome regardless
of the absolute scale of their feature vectors.

DBSCAN parameters
-----------------
    eps         Cosine distance below which two points are considered
                neighbours.  Default: 0.15 (points sharing drug + event
                will typically have cosine distance < 0.1).
    min_samples Minimum points in a neighbourhood to form a core point.
                Default: 2 (low, because pharmacovigilance data can have
                small but genuine clusters).

Both are exposed as parameters so callers can tune them.

Public API
----------
    cluster_reports(df, eps, min_samples)  →  list[ClusterResult]
    build_feature_matrix(df)               →  (matrix, feature_names)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MultiLabelBinarizer

from core.schemas import OUTCOME_CODES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default DBSCAN parameters
# ---------------------------------------------------------------------------

DEFAULT_EPS: float = 0.15          # cosine distance threshold
DEFAULT_MIN_SAMPLES: int = 2       # minimum cluster density

# Maximum number of top items shown in dominant_events / dominant_outcomes
_TOP_N: int = 3


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """
    Summary of a single DBSCAN cluster.

    Fields
    ------
    cluster_id : int
        DBSCAN label.  -1 means noise (reports that do not belong to any
        dense cluster).  All other values are ≥ 0.

    is_noise : bool
        True when cluster_id == -1.  Noise clusters should be reported
        separately from genuine clusters.

    size : int
        Number of reports in this cluster.

    dominant_drug : str
        The most frequently appearing drug_name in the cluster.
        Ties broken by first occurrence in sorted order.

    dominant_events : list[str]
        Top-N adverse event terms by frequency, most common first.

    dominant_outcomes : list[str]
        Top-N outcome codes by frequency, each rendered as
        "CODE (human label)", e.g. "HO (Hospitalisation)".

    drug_purity : float
        Fraction of reports in the cluster whose drug_name equals
        dominant_drug.  Range [0, 1].  A value of 1.0 means every
        report in the cluster involves the same drug.

    event_purity : float
        Fraction of reports whose adverse_event equals the most common
        adverse event.  Measures how focused the cluster is on one event.

    label : str
        Short auto-generated plain-English description, e.g.
        "warfarin / bleeding, nausea (n=45, purity=0.89)".
        Suitable for display in the UI without further processing.

    report_ids : list[str]
        IDs of all reports in this cluster, for traceability back to
        the source data.
    """

    cluster_id: int
    is_noise: bool
    size: int
    dominant_drug: str
    dominant_events: list[str]
    dominant_outcomes: list[str]
    drug_purity: float
    event_purity: float
    label: str
    report_ids: list[str]

    def to_dict(self) -> dict:
        """Serialise to a plain dict for API responses."""
        return {
            "cluster_id": self.cluster_id,
            "is_noise": self.is_noise,
            "size": self.size,
            "dominant_drug": self.dominant_drug,
            "dominant_events": self.dominant_events,
            "dominant_outcomes": self.dominant_outcomes,
            "drug_purity": round(self.drug_purity, 4),
            "event_purity": round(self.event_purity, 4),
            "label": self.label,
        }


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Convert a normalised FAERS DataFrame into a binary feature matrix.

    Each row in ``df`` becomes one row in the output matrix.
    Features are binary indicator columns for each unique value of
    ``drug_name``, ``adverse_event``, and ``outcome_code``.

    Parameters
    ----------
    df : pd.DataFrame
        Normalised FAERS DataFrame (output of faers_ingestor.load_faers).
        Must have columns: drug_name, adverse_event, outcome_code.

    Returns
    -------
    matrix : np.ndarray, shape (n_reports, n_features)
        Sparse binary matrix.  dtype float32 to reduce memory and speed
        up cosine distance computation.
    feature_names : list[str]
        Column labels for the matrix, e.g.
        ["drug:warfarin", "drug:ibuprofen", "event:bleeding", ...].

    Raises
    ------
    ValueError
        If required columns are absent from df.
    """
    required = {"drug_name", "adverse_event", "outcome_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_feature_matrix: missing columns {missing}")

    if df.empty:
        return np.empty((0, 0), dtype=np.float32), []

    # ── One-hot encode each categorical field separately ──────────────────
    def _onehot(series: pd.Series, prefix: str) -> tuple[pd.DataFrame, list[str]]:
        dummies = pd.get_dummies(
            series.fillna("UNKNOWN"), prefix=prefix, prefix_sep=":", dtype=np.float32
        )
        return dummies, list(dummies.columns)

    drug_dummies,    drug_cols    = _onehot(df["drug_name"],     "drug")
    event_dummies,   event_cols   = _onehot(df["adverse_event"], "event")
    outcome_dummies, outcome_cols = _onehot(df["outcome_code"],  "outcome")

    matrix = pd.concat([drug_dummies, event_dummies, outcome_dummies], axis=1)
    feature_names = drug_cols + event_cols + outcome_cols

    return matrix.values.astype(np.float32), feature_names


# ---------------------------------------------------------------------------
# Cluster statistics helper
# ---------------------------------------------------------------------------

def _top_n(series: pd.Series, n: int = _TOP_N) -> list[str]:
    """Return the top-n most frequent values in a Series, most common first."""
    return series.value_counts().head(n).index.tolist()


def _format_outcomes(outcome_list: list[str]) -> list[str]:
    """
    Convert raw outcome codes to human-readable strings.

    "HO" → "HO (Hospitalisation)"
    Unknown codes are passed through unchanged.
    """
    return [
        f"{code} ({OUTCOME_CODES.get(code, code)})" for code in outcome_list
    ]


def _compute_cluster_stats(cluster_df: pd.DataFrame, cluster_id: int) -> ClusterResult:
    """
    Compute summary statistics for one cluster subset.

    Parameters
    ----------
    cluster_df : pd.DataFrame
        Subset of the full FAERS DataFrame belonging to this cluster.
    cluster_id : int
        DBSCAN label for this cluster.

    Returns
    -------
    ClusterResult
    """
    size = len(cluster_df)
    is_noise = bool(cluster_id == -1)

    # ── Dominant drug ─────────────────────────────────────────────────────
    drug_counts = cluster_df["drug_name"].value_counts()
    dominant_drug = drug_counts.index[0] if len(drug_counts) > 0 else "unknown"
    drug_purity = float(drug_counts.iloc[0] / size) if size > 0 else 0.0

    # ── Dominant events ───────────────────────────────────────────────────
    event_counts = cluster_df["adverse_event"].value_counts()
    dominant_events = event_counts.head(_TOP_N).index.tolist()
    event_purity = float(event_counts.iloc[0] / size) if (size > 0 and len(event_counts) > 0) else 0.0

    # ── Dominant outcomes ─────────────────────────────────────────────────
    raw_outcomes = _top_n(cluster_df["outcome_code"])
    dominant_outcomes = _format_outcomes(raw_outcomes)

    # ── Report IDs ────────────────────────────────────────────────────────
    report_ids = cluster_df["report_id"].tolist() if "report_id" in cluster_df.columns else []

    # ── Auto-generated plain-English label ───────────────────────────────
    events_str = ", ".join(dominant_events[:2])  # max 2 events in label
    if is_noise:
        label = f"[noise] {size} unclustered report(s)"
    else:
        label = (
            f"{dominant_drug} / {events_str} "
            f"(n={size}, purity={drug_purity:.2f})"
        )

    return ClusterResult(
        cluster_id=cluster_id,
        is_noise=is_noise,
        size=size,
        dominant_drug=dominant_drug,
        dominant_events=dominant_events,
        dominant_outcomes=dominant_outcomes,
        drug_purity=drug_purity,
        event_purity=event_purity,
        label=label,
        report_ids=[str(r) for r in report_ids],
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def cluster_reports(
    df: pd.DataFrame,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> list[ClusterResult]:
    """
    Cluster FAERS adverse-event reports using DBSCAN with cosine distance.

    Parameters
    ----------
    df : pd.DataFrame
        Normalised FAERS DataFrame (output of faers_ingestor.load_faers).
        Must have columns: drug_name, adverse_event, outcome_code, report_id.

    eps : float
        DBSCAN neighbourhood radius in cosine distance space.
        Default: 0.15.  Smaller values → more, tighter clusters.
        Larger values → fewer, broader clusters.

    min_samples : int
        Minimum number of reports in a neighbourhood to form a core point.
        Default: 2.  Increase for noisier data.

    Returns
    -------
    list[ClusterResult]
        One ClusterResult per discovered cluster, including a noise cluster
        (cluster_id=-1) when any noise points exist.  Sorted by size
        descending, with the noise cluster always last.

    Notes
    -----
    Empty DataFrames return an empty list.
    DataFrames with fewer than min_samples rows will produce only noise.

    The algorithm runs in O(n²) time in the worst case.  For the bundled
    sample dataset (~300 rows) this takes < 10 ms.  For full FAERS quarterly
    data (~2 M rows), use the DuckDB aggregation path and sample before
    clustering.
    """
    if df.empty:
        logger.info("cluster_reports: empty DataFrame, returning []")
        return []

    # ── Build feature matrix ──────────────────────────────────────────────
    matrix, feature_names = build_feature_matrix(df)
    n_features = matrix.shape[1]

    logger.debug(
        "cluster_reports: %d reports × %d features", len(df), n_features
    )

    # ── Run DBSCAN ────────────────────────────────────────────────────────
    # metric='cosine' requires algorithm='brute' (no kd-tree for cosine).
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", algorithm="brute")
    labels: np.ndarray = dbscan.fit_predict(matrix)

    n_clusters = len(set(labels) - {-1})
    n_noise    = int((labels == -1).sum())

    logger.info(
        "cluster_reports: found %d cluster(s), %d noise point(s) "
        "(eps=%.3f, min_samples=%d)",
        n_clusters, n_noise, eps, min_samples,
    )

    # ── Attach labels to DataFrame for stat computation ───────────────────
    labelled = df.copy()
    labelled["_cluster_id"] = labels

    # ── Compute per-cluster statistics ────────────────────────────────────
    results: list[ClusterResult] = []
    noise_result: Optional[ClusterResult] = None

    for cluster_id in sorted(set(labels)):
        subset = labelled[labelled["_cluster_id"] == cluster_id].copy()
        result = _compute_cluster_stats(subset, cluster_id)

        if cluster_id == -1:
            noise_result = result  # hold back; append last
        else:
            results.append(result)

    # Sort genuine clusters by size descending
    results.sort(key=lambda r: r.size, reverse=True)

    # Append noise cluster at the end if it exists
    if noise_result is not None:
        results.append(noise_result)

    return results
