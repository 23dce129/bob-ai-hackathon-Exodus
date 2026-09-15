"""
priority_scorer.py — Safety Signal Priority Engine.

⚠️  DISCLAIMER ⚠️
This is a hackathon screening and prioritisation score only.
It is NOT an FDA-approved or regulatory risk score.
It is NOT a substitute for expert pharmacovigilance review.
It does NOT establish causality between a drug and an adverse event.
All flagged signals require clinical assessment by a qualified pharmacovigilance professional.

Purpose
-------
Combine six independent evidence dimensions into a single integer priority score
so that a reviewer can quickly rank signals that most warrant clinical attention.
The score is entirely deterministic and arithmetical — no language model is used.

Scoring dimensions and default weights
---------------------------------------
  1. PRR strength           (max 30 pts)
  2. Report volume          (max 20 pts)
  3. Statistical strength   (max 20 pts)
  4. Serious outcome rate   (max 15 pts)
  5. Reporting trend        (max 10 pts)
  6. Cluster strength       (max  5 pts)
                            ──────────
  Total                     (max 100 pts)

Each component produces a value in [0, 1] which is then multiplied by its
weight.  Final score = round(sum of weighted components).

Component formulas (all deterministic, no LLM)
-----------------------------------------------

1. PRR strength
   raw  = PRR point estimate from prr_calculator
   norm = min(raw / prr_saturation, 1.0)   [saturates at prr_saturation, default 20]
   pts  = norm × weight_prr

2. Report volume
   raw  = case_count (field `a` in the contingency table)
   norm = min(raw / volume_saturation, 1.0)  [saturates at volume_saturation, default 100]
   pts  = norm × weight_volume

3. Statistical strength
   raw  = p-value from chi-square or Fisher test
   norm = linear interpolation between pval_full (→ 1.0) and pval_none (→ 0.0):
          if p ≤ pval_full:   norm = 1.0
          if p ≥ pval_none:   norm = 0.0
          else:               norm = (pval_none - p) / (pval_none - pval_full)
   pts  = norm × weight_statistical

4. Serious outcome rate
   raw  = fraction of reports for this pair with a serious outcome code
          (DE=death, LT=life-threatening, HO=hospitalisation, DS=disability,
           CA=congenital anomaly, RI=required intervention)
   norm = min(raw / serious_saturation, 1.0)  [saturates at serious_saturation, default 0.50]
   pts  = norm × weight_serious

5. Reporting trend
   raw  = trend_score from temporal_analyser ∈ [-1.0, +1.0]
   multiplier = trend_direction_bonus[direction]
                  "increasing"        → 1.0
                  "stable"            → 0.5
                  "decreasing"        → 0.0
                  "insufficient_data" → 0.5  (neutral — no penalty for missing dates)
   pts  = multiplier × weight_trend

6. Cluster strength
   raw  = drug_purity of the best-matching cluster (highest drug_purity where
          the cluster's dominant_drug matches this signal's drug_name).
          0.0 if no matching cluster is found.
   norm = raw  (already in [0, 1])
   pts  = norm × weight_cluster

Configuration
-------------
All weights and saturation thresholds are held in ScoringConfig (schemas.py).
Pass a custom ScoringConfig to score_signal() to override any value.

Public API
----------
    score_signal(signal, df, trend, clusters, config)   → PriorityScore
    score_all_signals(signals, df, config)               → list[PriorityScore]
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from core.schemas import (
    OUTCOME_CODES,
    TREND_INSUFFICIENT_DATA,
    ComponentScore,
    PriorityScore,
    SignalResult,
    ScoringConfig,
    TrendResult,
)
from core.signal_clusterer import ClusterResult

logger = logging.getLogger(__name__)

# Outcome codes counted as "serious" for the serious-outcomes component
_SERIOUS_CODES = frozenset({"DE", "LT", "HO", "DS", "CA", "RI"})


# ---------------------------------------------------------------------------
# Individual component scorers (each returns a ComponentScore)
# ---------------------------------------------------------------------------

def _score_prr(prr: Optional[float], weight: int, saturation: float) -> ComponentScore:
    """
    PRR component.

    Formula: earned = min(prr / saturation, 1.0) × weight
    Full score when PRR ≥ saturation (default 20.0).
    Zero when PRR is None (undefined) or 0.
    """
    if prr is None or prr <= 0:
        return ComponentScore(
            name="PRR",
            earned=0.0,
            maximum=weight,
            raw_value=prr,
            formula=f"PRR is undefined or zero → 0/{weight}.",
        )
    norm = min(prr / saturation, 1.0)
    earned = norm * weight
    formula = (
        f"min(PRR={prr:.2f} / saturation={saturation:.1f}, 1.0) × {weight} "
        f"= {norm:.3f} × {weight} = {earned:.2f}"
    )
    return ComponentScore(
        name="PRR",
        earned=earned,
        maximum=weight,
        raw_value=prr,
        formula=formula,
    )


def _score_volume(case_count: int, weight: int, saturation: int) -> ComponentScore:
    """
    Report volume component.

    Formula: earned = min(case_count / saturation, 1.0) × weight
    Full score when case_count ≥ saturation (default 100 reports).
    """
    norm = min(case_count / saturation, 1.0) if saturation > 0 else 0.0
    earned = norm * weight
    formula = (
        f"min({case_count} reports / saturation={saturation}, 1.0) × {weight} "
        f"= {norm:.3f} × {weight} = {earned:.2f}"
    )
    return ComponentScore(
        name="Volume",
        earned=earned,
        maximum=weight,
        raw_value=float(case_count),
        formula=formula,
    )


def _score_statistical(
    p_value: Optional[float],
    weight: int,
    pval_full: float,
    pval_none: float,
) -> ComponentScore:
    """
    Statistical strength component.

    Formula: linear interpolation between pval_full (→ 1.0) and pval_none (→ 0.0).
        if p ≤ pval_full:   norm = 1.0
        if p ≥ pval_none:   norm = 0.0
        else:               norm = (pval_none - p) / (pval_none - pval_full)
    Full score when p ≤ 0.001 (default); zero score when p ≥ 0.05 (default).
    """
    if p_value is None:
        return ComponentScore(
            name="Statistical strength",
            earned=0.0,
            maximum=weight,
            raw_value=None,
            formula=f"p-value unavailable → 0/{weight}.",
        )
    if p_value <= pval_full:
        norm = 1.0
    elif p_value >= pval_none:
        norm = 0.0
    else:
        norm = (pval_none - p_value) / (pval_none - pval_full)

    earned = norm * weight
    formula = (
        f"Linear interpolation: p={p_value:.5f} between "
        f"pval_full={pval_full} (→1.0) and pval_none={pval_none} (→0.0) "
        f"→ norm={norm:.3f} × {weight} = {earned:.2f}"
    )
    return ComponentScore(
        name="Statistical strength",
        earned=earned,
        maximum=weight,
        raw_value=p_value,
        formula=formula,
    )


def _score_serious_outcomes(
    df: pd.DataFrame,
    drug: str,
    event: str,
    weight: int,
    saturation: float,
) -> ComponentScore:
    """
    Serious outcome rate component.

    Formula: earned = min(serious_rate / saturation, 1.0) × weight
    serious_rate = (reports with serious outcome) / (total reports for this pair)
    Serious outcomes: DE (Death), LT (Life-threatening), HO (Hospitalisation),
                      DS (Disability), CA (Congenital anomaly), RI (Required intervention).
    """
    mask = (df["drug_name"] == drug) & (df["adverse_event"] == event)
    subset = df.loc[mask]
    total = len(subset)

    if total == 0:
        return ComponentScore(
            name="Serious outcomes",
            earned=0.0,
            maximum=weight,
            raw_value=0.0,
            formula=f"No reports found for {drug}/{event} → 0/{weight}.",
        )

    serious_count = int(subset["outcome_code"].isin(_SERIOUS_CODES).sum())
    serious_rate = serious_count / total
    norm = min(serious_rate / saturation, 1.0) if saturation > 0 else 0.0
    earned = norm * weight
    formula = (
        f"Serious outcomes: {serious_count}/{total} reports "
        f"= rate {serious_rate:.3f}; "
        f"min(rate / saturation={saturation:.2f}, 1.0) × {weight} "
        f"= {norm:.3f} × {weight} = {earned:.2f}. "
        f"Serious codes: {', '.join(sorted(_SERIOUS_CODES))}."
    )
    return ComponentScore(
        name="Serious outcomes",
        earned=earned,
        maximum=weight,
        raw_value=round(serious_rate, 6),
        formula=formula,
    )


def _score_trend(
    trend: Optional[TrendResult],
    weight: int,
    direction_bonus: dict,
) -> ComponentScore:
    """
    Reporting trend component.

    Formula: earned = trend_direction_bonus[direction] × weight
    Bonus values:
        "increasing"        → 1.0 (full score — signal is growing)
        "stable"            → 0.5 (half score — signal persists but not growing)
        "decreasing"        → 0.0 (zero — signal appears to be resolving)
        "insufficient_data" → 0.5 (neutral — cannot penalise missing dates)
    """
    if trend is None:
        direction = TREND_INSUFFICIENT_DATA
        raw_val = None
        note = "No trend data available → neutral score."
    else:
        direction = trend.direction
        raw_val = trend.trend_score

    multiplier = direction_bonus.get(direction, 0.5)
    earned = multiplier * weight
    formula = (
        f"Direction='{direction}' → multiplier={multiplier:.1f} × {weight} = {earned:.2f}. "
        f"Raw trend_score={raw_val}."
        + (" " + note if trend is None else "")
    )
    return ComponentScore(
        name="Trend",
        earned=earned,
        maximum=weight,
        raw_value=raw_val,
        formula=formula,
    )


def _score_cluster(
    clusters: Optional[list],
    drug: str,
    weight: int,
) -> ComponentScore:
    """
    Cluster strength component.

    Formula: earned = drug_purity × weight
    Finds the highest-purity non-noise cluster whose dominant_drug matches `drug`.
    If no such cluster exists (e.g. drug only appears in the noise cluster),
    earned = 0.

    drug_purity is already in [0, 1], so no further normalisation is needed.
    A purity of 1.0 means every report in the cluster involves this drug —
    the cluster is a strong, focused signal.
    """
    if not clusters:
        return ComponentScore(
            name="Cluster strength",
            earned=0.0,
            maximum=weight,
            raw_value=None,
            formula=f"No cluster data provided → 0/{weight}.",
        )

    best_purity = 0.0
    for cluster in clusters:
        if cluster.is_noise:
            continue
        if cluster.dominant_drug == drug:
            best_purity = max(best_purity, cluster.drug_purity)

    earned = best_purity * weight
    formula = (
        f"Best non-noise cluster purity for '{drug}' = {best_purity:.4f}; "
        f"purity × {weight} = {earned:.2f}."
    )
    return ComponentScore(
        name="Cluster strength",
        earned=earned,
        maximum=weight,
        raw_value=round(best_purity, 6),
        formula=formula,
    )


# ---------------------------------------------------------------------------
# Public entry point — single signal
# ---------------------------------------------------------------------------

def score_signal(
    signal: SignalResult,
    df: pd.DataFrame,
    trend: Optional[TrendResult] = None,
    clusters: Optional[list] = None,
    config: Optional[ScoringConfig] = None,
) -> PriorityScore:
    """
    Compute the priority score for one flagged signal.

    Parameters
    ----------
    signal : SignalResult
        Output of prr_calculator.calculate_all_signals().
    df : pd.DataFrame
        Normalised FAERS DataFrame (for serious-outcome rate calculation).
    trend : TrendResult, optional
        Output of temporal_analyser.analyse_trend() for this pair.
        Pass None to use the neutral score (0.5 × weight_trend).
    clusters : list[ClusterResult], optional
        Output of signal_clusterer.cluster_reports().
        Pass None or [] to score the cluster component as 0.
    config : ScoringConfig, optional
        Override default weights and thresholds.
        Defaults to ScoringConfig() if not supplied.

    Returns
    -------
    PriorityScore
    """
    if config is None:
        config = ScoringConfig()

    drug  = signal.drug_name
    event = signal.adverse_event

    components = [
        _score_prr(
            signal.prr,
            config.weight_prr,
            config.prr_saturation,
        ),
        _score_volume(
            signal.a,
            config.weight_volume,
            config.volume_saturation,
        ),
        _score_statistical(
            signal.p_value,
            config.weight_statistical,
            config.pval_full,
            config.pval_none,
        ),
        _score_serious_outcomes(
            df,
            drug,
            event,
            config.weight_serious,
            config.serious_saturation,
        ),
        _score_trend(
            trend,
            config.weight_trend,
            config.trend_direction_bonus,
        ),
        _score_cluster(
            clusters,
            drug,
            config.weight_cluster,
        ),
    ]

    final_score = min(
        round(sum(c.earned for c in components)),
        config.max_score,
    )

    logger.debug(
        "score_signal: %s/%s → %d/%d  components=%s",
        drug, event, final_score, config.max_score,
        [f"{c.name}={c.earned:.1f}" for c in components],
    )

    return PriorityScore(
        drug_name=drug,
        adverse_event=event,
        final_score=final_score,
        components=components,
        config=config,
    )


# ---------------------------------------------------------------------------
# Public entry point — all signals
# ---------------------------------------------------------------------------

def score_all_signals(
    signals: list[SignalResult],
    df: pd.DataFrame,
    trends: Optional[dict] = None,
    clusters: Optional[list] = None,
    config: Optional[ScoringConfig] = None,
) -> list[PriorityScore]:
    """
    Compute priority scores for a list of SignalResults.

    Parameters
    ----------
    signals : list[SignalResult]
        Typically the flagged subset from calculate_all_signals().
    df : pd.DataFrame
        Normalised FAERS DataFrame.
    trends : dict, optional
        Mapping of "{drug_name}|{adverse_event}" → TrendResult.
        Build with: {f"{t.drug_name}|{t.adverse_event}": t for t in analyse_all_trends(df)}
    clusters : list[ClusterResult], optional
        Output of cluster_reports(df).
    config : ScoringConfig, optional

    Returns
    -------
    list[PriorityScore]
        Sorted by final_score descending (highest priority first).
    """
    if config is None:
        config = ScoringConfig()

    results: list[PriorityScore] = []
    for sig in signals:
        key = f"{sig.drug_name}|{sig.adverse_event}"
        trend = (trends or {}).get(key)
        results.append(score_signal(sig, df, trend, clusters, config))

    results.sort(key=lambda ps: ps.final_score, reverse=True)
    return results
