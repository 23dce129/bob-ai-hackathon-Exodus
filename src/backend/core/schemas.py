"""
schemas.py — Internal data schemas for PharmaGuard AI.

These are the shared contracts between the ingestion layer (faers_ingestor)
and the statistical layer (prr_calculator).  No file I/O happens here.

Pydantic models are used for validation when receiving data from external
sources (API requests).  Plain dataclasses are used for internal pipeline
results where speed matters more than validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

# ---------------------------------------------------------------------------
# Disclaimer — embedded in every SignalResult so it cannot be omitted.
# ---------------------------------------------------------------------------
_PRR_CAUTION = (
    "PRR is a statistical association measure. "
    "It does not establish causality between the drug and the adverse event."
)

# ---------------------------------------------------------------------------
# Normalised adverse-event report (output of faers_ingestor)
# ---------------------------------------------------------------------------

# Required columns in the raw FAERS CSV after normalisation.
# The ingestor maps vendor-specific names to these standard names.
REQUIRED_INTERNAL_COLUMNS: list[str] = [
    "report_id",
    "drug_name",
    "adverse_event",
    "outcome_code",
    "report_date",
    "age_years",
    "sex",
]

# Outcome codes used by FDA FAERS
OUTCOME_CODES: dict[str, str] = {
    "DE": "Death",
    "LT": "Life-threatening",
    "HO": "Hospitalisation",
    "DS": "Disability",
    "CA": "Congenital anomaly",
    "RI": "Required intervention",
    "OT": "Other serious",
}


# ---------------------------------------------------------------------------
# 2×2 contingency table
# ---------------------------------------------------------------------------

@dataclass
class ContingencyTable:
    """
    2×2 table for a specific (drug, event) pair.

    Layout:
                     Event E    Not Event E
        Drug D           a            b
        Not Drug D       c            d

    All counts are non-negative integers.  The calculator checks for
    zero-denominator conditions before computing PRR.
    """

    drug: str
    event: str
    a: int   # drug D with event E
    b: int   # drug D without event E
    c: int   # other drugs with event E
    d: int   # other drugs without event E

    @property
    def n(self) -> int:
        """Total number of reports in the dataset."""
        return self.a + self.b + self.c + self.d

    @property
    def drug_total(self) -> int:
        """All reports involving drug D."""
        return self.a + self.b

    @property
    def event_total(self) -> int:
        """All reports involving event E."""
        return self.a + self.c

    def __post_init__(self) -> None:
        for attr in ("a", "b", "c", "d"):
            if getattr(self, attr) < 0:
                raise ValueError(f"ContingencyTable.{attr} must be ≥ 0, got {getattr(self, attr)}")


# ---------------------------------------------------------------------------
# PRR result
# ---------------------------------------------------------------------------

@dataclass
class PRRResult:
    """
    Point estimate and 95 % confidence interval for Proportional Reporting Ratio.

    Formula
    -------
        PRR = [a / (a+b)] / [c / (c+d)]

        SE  = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
        95% CI lower = exp(ln(PRR) - 1.96 × SE)
        95% CI upper = exp(ln(PRR) + 1.96 × SE)

    All three values are None when the denominator is zero (c = 0 or a = 0),
    which means the calculation is mathematically undefined.
    """

    prr: Optional[float]
    ci_lower_95: Optional[float]
    ci_upper_95: Optional[float]
    calculable: bool   # False when any denominator is zero


# ---------------------------------------------------------------------------
# Chi-square / Fisher result
# ---------------------------------------------------------------------------

@dataclass
class Chi2Result:
    """
    Result of the independence test on the 2×2 contingency table.

    When any expected cell count is < 5, Fisher's exact test is used instead
    of the chi-square approximation (the `method` field records which was used).
    """

    statistic: Optional[float]   # chi2 statistic (None for Fisher)
    p_value: float
    method: str   # "chi2" | "fisher"


# ---------------------------------------------------------------------------
# Final signal result (one row in the output)
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    """
    Complete result for one (drug, adverse_event) pair.

    Fields
    ------
    drug_name, adverse_event
        Normalised strings from the FAERS data.

    a, b, c, d
        Raw 2×2 table counts — always present so consumers can verify the maths.

    prr, ci_lower_95, ci_upper_95
        None when PRR is mathematically undefined (see PRRResult).

    chi2_statistic, p_value, test_method
        From Chi2Result.

    signal_flag
        True only when all three EMA/WHO criteria are simultaneously met:
            a ≥ min_cases  AND  PRR ≥ min_prr  AND  p_value ≤ max_pval

    flag_reason
        Plain-English explanation of why this pair was or was not flagged.
        Never empty.  Example:
            "Flagged: PRR=4.2 (≥2.0 threshold), 7 cases (≥3 threshold), p=0.003."
            "Not flagged: only 2 cases reported (minimum 3 required)."

    caution
        Always set to the module-level _PRR_CAUTION disclaimer.
        Cannot be overridden by callers.
    """

    drug_name: str
    adverse_event: str

    # Contingency table cells
    a: int
    b: int
    c: int
    d: int

    # PRR
    prr: Optional[float]
    ci_lower_95: Optional[float]
    ci_upper_95: Optional[float]

    # Independence test
    chi2_statistic: Optional[float]
    p_value: Optional[float]
    test_method: str   # "chi2" | "fisher" | "undetermined"

    # Interpretation
    signal_flag: bool
    flag_reason: str

    # Immutable disclaimer — set automatically, cannot be changed after creation
    caution: str = field(default=_PRR_CAUTION, init=False)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for API responses and JSON export)."""
        return {
            "drug_name": self.drug_name,
            "adverse_event": self.adverse_event,
            "contingency": {"a": self.a, "b": self.b, "c": self.c, "d": self.d},
            "prr": self.prr,
            "ci_lower_95": self.ci_lower_95,
            "ci_upper_95": self.ci_upper_95,
            "chi2_statistic": self.chi2_statistic,
            "p_value": self.p_value,
            "test_method": self.test_method,
            "signal_flag": self.signal_flag,
            "flag_reason": self.flag_reason,
            "caution": self.caution,
        }


# ---------------------------------------------------------------------------
# Temporal trend result
# ---------------------------------------------------------------------------

# Minimum number of quarters with data required before we report a trend.
# With fewer than this, trend direction is meaningless.
TREND_MIN_QUARTERS: int = 2

# Labels used in TrendResult.direction — kept as module constants so that
# consumers can compare with these rather than raw strings.
TREND_INCREASING = "increasing"
TREND_DECREASING = "decreasing"
TREND_STABLE     = "stable"
TREND_INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class QuarterCount:
    """Report count for one calendar quarter."""
    quarter: str        # ISO format "YYYY-Qq", e.g. "2023-Q1"
    year: int
    quarter_num: int    # 1–4
    count: int

    def to_dict(self) -> dict:
        return {
            "quarter": self.quarter,
            "year": self.year,
            "quarter_num": self.quarter_num,
            "count": self.count,
        }


@dataclass
class TrendResult:
    """
    Temporal trend analysis for one (drug, adverse_event) pair.

    Fields
    ------
    drug_name, adverse_event
        The pair being analysed.

    quarters : list[QuarterCount]
        Report counts per calendar quarter, sorted oldest → newest.
        Quarters with zero reports are included only when they fall
        within the observed date range (no leading/trailing zero-padding).

    recent_count : int
        Sum of reports in the most recent half of the observed quarters.

    previous_count : int
        Sum of reports in the older half of the observed quarters.

    pct_change : float | None
        Percentage change from previous_count to recent_count.
        Formula: ((recent - previous) / previous) × 100
        None when previous_count == 0 (avoids division by zero).

    trend_score : float
        Deterministic numerical score summarising the trend direction.
        Computed as:
            score = (recent_rate - previous_rate) / max_rate
        where rate = count / n_quarters_in_period, max_rate = max(rates).
        Range: [-1.0, +1.0].  Positive = increasing, negative = decreasing.
        0.0 when data is insufficient or rates are equal.

    direction : str
        One of: "increasing" | "decreasing" | "stable" | "insufficient_data".
        Thresholds (applied to trend_score):
            score > +STABLE_BAND  → "increasing"
            score < -STABLE_BAND  → "decreasing"
            otherwise             → "stable"
        "insufficient_data" when fewer than TREND_MIN_QUARTERS quarters
        have any reports.

    total_reports : int
        Total reports for this pair across all quarters.

    date_range_start : str | None
        Earliest report date observed (ISO format), or None if no dates.

    date_range_end : str | None
        Latest report date observed (ISO format), or None if no dates.
    """

    drug_name: str
    adverse_event: str
    quarters: list[QuarterCount]
    recent_count: int
    previous_count: int
    pct_change: Optional[float]       # None when previous_count == 0
    trend_score: float                # [-1.0, +1.0]
    direction: str
    total_reports: int
    date_range_start: Optional[str]
    date_range_end: Optional[str]

    def to_dict(self) -> dict:
        return {
            "drug_name": self.drug_name,
            "adverse_event": self.adverse_event,
            "quarters": [q.to_dict() for q in self.quarters],
            "recent_count": self.recent_count,
            "previous_count": self.previous_count,
            "pct_change": round(self.pct_change, 2) if self.pct_change is not None else None,
            "trend_score": round(self.trend_score, 4),
            "direction": self.direction,
            "total_reports": self.total_reports,
            "date_range_start": self.date_range_start,
            "date_range_end": self.date_range_end,
        }


# ---------------------------------------------------------------------------
# Priority scoring — configurable weights and output schema
# ---------------------------------------------------------------------------

# ── Component maximum points (must sum to 100) ────────────────────────────
# These are the defaults.  Callers can pass a ScoringConfig to override.
# The values below match the example in the product brief:
#   PRR 30 + Volume 20 + Statistical 20 + Serious outcomes 15 + Trend 10 + Cluster 5 = 100
DEFAULT_WEIGHT_PRR        = 30
DEFAULT_WEIGHT_VOLUME     = 20
DEFAULT_WEIGHT_STATISTICAL= 20
DEFAULT_WEIGHT_SERIOUS    = 15
DEFAULT_WEIGHT_TREND      = 10
DEFAULT_WEIGHT_CLUSTER    =  5

# ── PRR saturation point ─────────────────────────────────────────────────
# PRR values above this threshold receive the full PRR component score.
# Chosen as 10× the EMA flagging threshold (2.0 × 10 = 20).
DEFAULT_PRR_SATURATION    = 20.0

# ── Volume saturation point ───────────────────────────────────────────────
# Case counts above this receive the full volume component score.
# Set to 100: a dataset that contributes ≥100 co-reports has strong evidence.
DEFAULT_VOLUME_SATURATION = 100

# ── Statistical strength: p-value mapping ────────────────────────────────
# Full score at p ≤ 0.001; zero score at p ≥ 0.05.
DEFAULT_PVAL_FULL    = 0.001
DEFAULT_PVAL_NONE    = 0.05

# ── Serious outcome rate saturation ──────────────────────────────────────
# Fraction of reports with a serious outcome (DE/LT/HO/DS/CA/RI) that
# receives the full serious-outcomes component score.
DEFAULT_SERIOUS_SATURATION = 0.50   # ≥50 % serious → full score

# ── Trend direction multipliers ──────────────────────────────────────────
# Applied to the raw trend_score ∈ [-1, +1] to map it onto [0, 1].
# "increasing" gets the maximum; "decreasing" gets the minimum.
# Formula: component = weight × trend_multiplier(direction, trend_score)
DEFAULT_TREND_DIRECTION_BONUS = {
    "increasing":        1.0,
    "stable":            0.5,
    "decreasing":        0.0,
    "insufficient_data": 0.5,   # neutral — no penalty for missing dates
}


@dataclass
class ScoringConfig:
    """
    Configurable weights and saturation thresholds for the priority scorer.

    All weight fields must sum to 100.  The scorer does not enforce this
    at construction time (to allow partial overrides in tests), but
    PriorityScore.final_score is always computed from the actual component
    scores rather than the weights directly, so a misconfigured sum will
    simply produce a score that does not top out at 100.

    Defaults match the example in the product brief:
        PRR 30 | Volume 20 | Statistical 20 | Serious 15 | Trend 10 | Cluster 5
    """
    weight_prr:          int   = DEFAULT_WEIGHT_PRR
    weight_volume:       int   = DEFAULT_WEIGHT_VOLUME
    weight_statistical:  int   = DEFAULT_WEIGHT_STATISTICAL
    weight_serious:      int   = DEFAULT_WEIGHT_SERIOUS
    weight_trend:        int   = DEFAULT_WEIGHT_TREND
    weight_cluster:      int   = DEFAULT_WEIGHT_CLUSTER

    prr_saturation:      float = DEFAULT_PRR_SATURATION
    volume_saturation:   int   = DEFAULT_VOLUME_SATURATION
    pval_full:           float = DEFAULT_PVAL_FULL
    pval_none:           float = DEFAULT_PVAL_NONE
    serious_saturation:  float = DEFAULT_SERIOUS_SATURATION
    trend_direction_bonus: dict = None   # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.trend_direction_bonus is None:
            self.trend_direction_bonus = dict(DEFAULT_TREND_DIRECTION_BONUS)

    @property
    def max_score(self) -> int:
        return (
            self.weight_prr + self.weight_volume + self.weight_statistical
            + self.weight_serious + self.weight_trend + self.weight_cluster
        )


@dataclass
class ComponentScore:
    """
    Score and explanation for one component of the priority score.

    Fields
    ------
    name : str
        Human-readable component name, e.g. "PRR".
    earned : float
        Points earned for this component.
    maximum : int
        Maximum possible points for this component (from ScoringConfig).
    raw_value : float | None
        The underlying measurement before scoring, e.g. the PRR value.
        None when the measurement was unavailable.
    formula : str
        One-sentence description of how earned was calculated from raw_value.
    """
    name: str
    earned: float
    maximum: int
    raw_value: Optional[float]
    formula: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "earned": round(self.earned, 2),
            "maximum": self.maximum,
            "raw_value": round(self.raw_value, 4) if self.raw_value is not None else None,
            "formula": self.formula,
        }

    def summary_line(self) -> str:
        """One-line display string matching the example in the product brief."""
        return f"{self.name}: {round(self.earned, 1)}/{self.maximum}"


@dataclass
class PriorityScore:
    """
    Final priority score for one (drug, adverse_event) pair.

    ⚠️  DISCLAIMER — printed on every instance ⚠️
    This is a hackathon screening and prioritisation score only.
    It is NOT an FDA-approved or regulatory risk score.
    It is NOT a substitute for clinical review.
    It does NOT establish causality between the drug and the adverse event.
    Flagged signals require expert pharmacovigilance assessment.

    Fields
    ------
    drug_name, adverse_event
        The pair being scored.
    final_score : int
        Integer in [0, config.max_score].  Sum of all component earned values,
        rounded to the nearest integer.
    components : list[ComponentScore]
        One entry per scoring dimension, in display order.
    config : ScoringConfig
        The configuration used to produce this score.  Always present so
        consumers can reproduce the calculation.
    disclaimer : str
        Always set to the module-level _PRIORITY_DISCLAIMER.
        Cannot be overridden by callers.
    """
    drug_name: str
    adverse_event: str
    final_score: int
    components: list[ComponentScore]
    config: ScoringConfig
    disclaimer: str = field(
        default=(
            "⚠️  HACKATHON SCREENING SCORE ONLY.  "
            "This score is NOT an FDA-approved or regulatory risk assessment.  "
            "It is NOT a substitute for expert pharmacovigilance review.  "
            "It does NOT establish causality.  "
            "For research and demonstration purposes only."
        ),
        init=False,
    )

    def display(self) -> str:
        """
        Multi-line human-readable display matching the product brief example.

        Example output:
            Signal score: 87

            Contributors:
            PRR: 29/30
            Volume: 16/20
            Statistical strength: 18/20
            Serious outcomes: 12/15
            Trend: 8/10
            Cluster strength: 4/5

            ⚠️  HACKATHON SCREENING SCORE ONLY. ...
        """
        lines = [
            f"Signal score: {self.final_score}",
            "",
            "Contributors:",
        ]
        for c in self.components:
            lines.append(f"  {c.summary_line()}")
        lines.append("")
        lines.append(self.disclaimer)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "drug_name":   self.drug_name,
            "adverse_event": self.adverse_event,
            "final_score": self.final_score,
            "max_possible_score": self.config.max_score,
            "components":  [c.to_dict() for c in self.components],
            "disclaimer":  self.disclaimer,
        }
