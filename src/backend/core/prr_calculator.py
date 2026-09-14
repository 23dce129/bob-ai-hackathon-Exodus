"""
prr_calculator.py — Proportional Reporting Ratio (PRR) signal detection.

This module takes a clean, normalised FAERS DataFrame (produced by
faers_ingestor.load_faers) and computes PRR statistics for every
drug–adverse-event pair.

Statistical background
----------------------
PRR is a disproportionality analysis method used by the European Medicines
Agency (EMA) and the WHO Uppsala Monitoring Centre (UMC) as a first-pass
pharmacovigilance signal-detection screen.

Given a drug D and an adverse event E, we build a 2×2 contingency table:

                     Event E    Not Event E
        Drug D           a            b
        Not Drug D       c            d

PRR point estimate
~~~~~~~~~~~~~~~~~~
    PRR = [a / (a+b)] / [c / (c+d)]

    Interpretation: a PRR of 3 means drug D is reported with event E
    three times more often (proportionally) than all other drugs combined.
    This is a measure of statistical association only.

95 % confidence interval (log-normal method, Evans 2001)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SE  = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    CI lower = exp(ln(PRR) - 1.96 × SE)
    CI upper = exp(ln(PRR) + 1.96 × SE)

Independence test
~~~~~~~~~~~~~~~~~
    chi-square test (scipy.stats.chi2_contingency) when all expected
    cell counts ≥ 5.  Fisher's exact test (scipy.stats.fisher_exact)
    otherwise.

EMA/WHO signal-flagging criteria (all three must hold simultaneously)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. a ≥ min_cases (default 3)   — evidence threshold
    2. PRR ≥ min_prr  (default 2.0) — effect-size threshold
    3. p-value ≤ max_pval (default 0.05) — statistical threshold

IMPORTANT DISCLAIMER
--------------------
    PRR is a statistical association measure.
    It does NOT establish causality between a drug and an adverse event.
    A flagged signal requires clinical review and further investigation.

References
----------
    Evans SJW et al. (2001). Use of proportional reporting ratios (PRRs)
    for signal generation from spontaneous adverse drug reaction reports.
    Pharmacoepidemiology and Drug Safety, 10(6):483–486.

    EMA (2006). Guideline on the use of statistical signal detection methods
    in the EudraVigilance data analysis system. EMEA/106464/2006.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from core.schemas import (
    Chi2Result,
    ContingencyTable,
    PRRResult,
    SignalResult,
)

logger = logging.getLogger(__name__)

# Default EMA/WHO thresholds
DEFAULT_MIN_CASES: int = 3
DEFAULT_MIN_PRR: float = 2.0
DEFAULT_MAX_PVAL: float = 0.05


# ---------------------------------------------------------------------------
# Contingency table
# ---------------------------------------------------------------------------

def build_contingency_table(
    df: pd.DataFrame, drug: str, event: str
) -> ContingencyTable:
    """
    Build the 2×2 contingency table for a specific (drug, event) pair.

    Parameters
    ----------
    df : pd.DataFrame
        Normalised FAERS DataFrame (from faers_ingestor.load_faers).
        Must have columns 'drug_name' and 'adverse_event'.
    drug : str
        Normalised drug name (must match values in df['drug_name']).
    event : str
        Normalised adverse event term (must match df['adverse_event']).

    Returns
    -------
    ContingencyTable
        Counts a, b, c, d as described in the module docstring.
    """
    is_drug = df["drug_name"] == drug
    is_event = df["adverse_event"] == event

    a = int((is_drug & is_event).sum())
    b = int((is_drug & ~is_event).sum())
    c = int((~is_drug & is_event).sum())
    d = int((~is_drug & ~is_event).sum())

    return ContingencyTable(drug=drug, event=event, a=a, b=b, c=c, d=d)


# ---------------------------------------------------------------------------
# PRR calculation
# ---------------------------------------------------------------------------

def calculate_prr(ct: ContingencyTable) -> PRRResult:
    """
    Compute the PRR point estimate and 95 % confidence interval.

    Returns a PRRResult with all three values set to None when the
    calculation is mathematically undefined (a=0 or c=0 or a+b=0 or c+d=0).

    Formula
    -------
        PRR = [a / (a+b)] / [c / (c+d)]
        SE  = sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
        95% CI = exp(ln(PRR) ± 1.96 × SE)

    Parameters
    ----------
    ct : ContingencyTable
        Pre-built 2×2 table.

    Returns
    -------
    PRRResult
    """
    a, b, c, d = ct.a, ct.b, ct.c, ct.d
    drug_total = a + b  # total reports for this drug
    other_total = c + d  # total reports for all other drugs

    # Guard: any zero denominator makes PRR undefined
    if a == 0 or c == 0 or drug_total == 0 or other_total == 0:
        return PRRResult(prr=None, ci_lower_95=None, ci_upper_95=None, calculable=False)

    prr = (a / drug_total) / (c / other_total)

    # Standard error of ln(PRR)
    try:
        se = math.sqrt(1 / a - 1 / drug_total + 1 / c - 1 / other_total)
    except ValueError:
        # Should not happen when a≥1 and c≥1, but guard anyway
        return PRRResult(prr=prr, ci_lower_95=None, ci_upper_95=None, calculable=True)

    ln_prr = math.log(prr)
    ci_lower = math.exp(ln_prr - 1.96 * se)
    ci_upper = math.exp(ln_prr + 1.96 * se)

    return PRRResult(
        prr=round(prr, 4),
        ci_lower_95=round(ci_lower, 4),
        ci_upper_95=round(ci_upper, 4),
        calculable=True,
    )


# ---------------------------------------------------------------------------
# Independence test
# ---------------------------------------------------------------------------

def calculate_chi2(ct: ContingencyTable) -> Chi2Result:
    """
    Test independence of drug and event using chi-square or Fisher's exact test.

    Selection rule:
        Use chi-square when all four *expected* cell counts are ≥ 5.
        Use Fisher's exact test otherwise (small sample correction).

    Parameters
    ----------
    ct : ContingencyTable

    Returns
    -------
    Chi2Result
        `method` is "chi2" or "fisher".
        `statistic` is None when Fisher's exact test is used (Fisher returns
        an odds ratio, not a chi2 statistic — they are not interchangeable).
    """
    table = [[ct.a, ct.b], [ct.c, ct.d]]

    # Compute expected counts to decide which test to use
    n = ct.n
    if n == 0:
        return Chi2Result(statistic=None, p_value=1.0, method="undetermined")

    expected_a = (ct.drug_total * ct.event_total) / n
    expected_b = (ct.drug_total * (n - ct.event_total)) / n
    expected_c = ((n - ct.drug_total) * ct.event_total) / n
    expected_d = ((n - ct.drug_total) * (n - ct.event_total)) / n

    if min(expected_a, expected_b, expected_c, expected_d) >= 5:
        chi2_stat, p_val, _dof, _expected = chi2_contingency(table, correction=False)
        return Chi2Result(
            statistic=round(float(chi2_stat), 4),
            p_value=round(float(p_val), 6),
            method="chi2",
        )
    else:
        # Fisher's exact test — two-sided
        _odds_ratio, p_val = fisher_exact(table, alternative="two-sided")
        return Chi2Result(
            statistic=None,
            p_value=round(float(p_val), 6),
            method="fisher",
        )


# ---------------------------------------------------------------------------
# Signal flagging
# ---------------------------------------------------------------------------

def flag_signal(
    ct: ContingencyTable,
    prr_result: PRRResult,
    chi2_result: Chi2Result,
    min_cases: int = DEFAULT_MIN_CASES,
    min_prr: float = DEFAULT_MIN_PRR,
    max_pval: float = DEFAULT_MAX_PVAL,
) -> tuple[bool, str]:
    """
    Apply EMA/WHO signal-flagging criteria and return (flagged, reason).

    All three criteria must hold for a signal to be flagged:
        1. case_count (a) ≥ min_cases
        2. PRR ≥ min_prr
        3. p_value ≤ max_pval

    The reason string is always a complete, human-readable sentence
    explaining which criteria were or were not met.  It never contains
    internal variable names or numeric codes.

    Parameters
    ----------
    ct : ContingencyTable
    prr_result : PRRResult
    chi2_result : Chi2Result
    min_cases : int
    min_prr : float
    max_pval : float

    Returns
    -------
    tuple[bool, str]
        (True, "Flagged: ...") or (False, "Not flagged: ...")
    """
    reasons_fail: list[str] = []

    # Criterion 1 — case count
    if ct.a < min_cases:
        reasons_fail.append(
            f"only {ct.a} case(s) reported (minimum {min_cases} required)"
        )

    # Criterion 2 — PRR calculable and above threshold
    if not prr_result.calculable or prr_result.prr is None:
        reasons_fail.append(
            "PRR could not be calculated (no reports of this event for other drugs)"
        )
    elif prr_result.prr < min_prr:
        reasons_fail.append(
            f"PRR={prr_result.prr:.2f} is below the {min_prr:.1f} threshold"
        )

    # Criterion 3 — statistical significance
    if chi2_result.p_value > max_pval:
        reasons_fail.append(
            f"p-value={chi2_result.p_value:.4f} exceeds the {max_pval} significance threshold"
        )

    if reasons_fail:
        return False, "Not flagged: " + "; ".join(reasons_fail) + "."

    # All criteria met
    return True, (
        f"Flagged: PRR={prr_result.prr:.2f} (≥{min_prr:.1f} threshold), "
        f"{ct.a} cases (≥{min_cases} threshold), "
        f"p={chi2_result.p_value:.4f} (≤{max_pval} threshold) "
        f"using {chi2_result.method} test."
    )


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def calculate_all_signals(
    df: pd.DataFrame,
    min_cases: int = DEFAULT_MIN_CASES,
    min_prr: float = DEFAULT_MIN_PRR,
    max_pval: float = DEFAULT_MAX_PVAL,
    drug_filter: Optional[str] = None,
) -> list[SignalResult]:
    """
    Compute PRR for every (drug, adverse_event) pair in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Normalised FAERS DataFrame.
    min_cases : int
        Minimum number of co-reports required (EMA default: 3).
    min_prr : float
        Minimum PRR point estimate required (EMA default: 2.0).
    max_pval : float
        Maximum p-value for statistical significance (default: 0.05).
    drug_filter : str, optional
        If supplied, only compute signals for this drug name.
        Useful for single-drug queries from the API.

    Returns
    -------
    list[SignalResult]
        All pairs, sorted by PRR descending (None PRR values last).
        Consumers can filter on `signal_flag` to get only flagged signals.

    Notes
    -----
    This function is O(drugs × events).  For the bundled sample dataset
    (~300 rows, 6 drugs, 10 events) it runs in milliseconds.  For full
    FAERS quarterly data (~2 million rows, thousands of drug–event pairs),
    use the DuckDB-backed aggregation path (future phase).
    """
    if df.empty:
        return []

    drugs = [drug_filter] if drug_filter else sorted(df["drug_name"].unique())
    events = sorted(df["adverse_event"].unique())

    results: list[SignalResult] = []

    for drug in drugs:
        for event in events:
            ct = build_contingency_table(df, drug, event)

            # Skip pairs where this drug has never reported this event at all
            if ct.a == 0:
                continue

            prr_result = calculate_prr(ct)
            chi2_result = calculate_chi2(ct)
            flagged, reason = flag_signal(
                ct, prr_result, chi2_result, min_cases, min_prr, max_pval
            )

            results.append(
                SignalResult(
                    drug_name=drug,
                    adverse_event=event,
                    a=ct.a,
                    b=ct.b,
                    c=ct.c,
                    d=ct.d,
                    prr=prr_result.prr,
                    ci_lower_95=prr_result.ci_lower_95,
                    ci_upper_95=prr_result.ci_upper_95,
                    chi2_statistic=chi2_result.statistic,
                    p_value=chi2_result.p_value,
                    test_method=chi2_result.method,
                    signal_flag=flagged,
                    flag_reason=reason,
                )
            )

    # Sort by PRR descending; pairs where PRR is None go to the end
    results.sort(
        key=lambda r: (r.prr is None, -(r.prr or 0.0))
    )

    flagged_count = sum(1 for r in results if r.signal_flag)
    logger.info(
        "calculate_all_signals: %d pairs evaluated, %d flagged "
        "(drug_filter=%r, min_cases=%d, min_prr=%.1f, max_pval=%.2f)",
        len(results), flagged_count, drug_filter, min_cases, min_prr, max_pval,
    )

    return results
