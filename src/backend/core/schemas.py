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
