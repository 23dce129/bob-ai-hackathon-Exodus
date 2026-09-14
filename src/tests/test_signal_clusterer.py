"""
test_signal_clusterer.py — Unit tests for DBSCAN-based adverse-event clustering.

Test strategy
-------------
DBSCAN is a deterministic algorithm given the same input matrix and parameters.
We exploit this by constructing synthetic DataFrames where the correct cluster
assignments are known by design:

    Cluster A  — reports all involving drugA / bleeding / HO
                 These will be very close in cosine space (distance ≈ 0)
                 because they share identical feature values.

    Cluster B  — reports all involving drugB / myalgia / OT
                 Completely different feature set → far from Cluster A.

    Noise      — a single isolated report with a unique drug/event/outcome
                 combination that appears only once and cannot form a core
                 point when min_samples=2.

By separating the clusters in feature space and controlling min_samples we
can assert exact cluster memberships without randomness.

What is NOT tested
------------------
- Specific DBSCAN internals (core-point reachability chains) — we test
  observable outputs only.
- Performance / scalability — out of scope for unit tests.
- eps sensitivity beyond the parameter-sweep test — too many combinations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.signal_clusterer import (
    DEFAULT_EPS,
    DEFAULT_MIN_SAMPLES,
    ClusterResult,
    _compute_cluster_stats,
    _format_outcomes,
    _top_n,
    build_feature_matrix,
    cluster_reports,
)


# ---------------------------------------------------------------------------
# Synthetic DataFrame factories
# ---------------------------------------------------------------------------

def _make_report(
    drug: str,
    event: str,
    outcome: str = "OT",
    n: int = 1,
    id_offset: int = 0,
) -> list[dict]:
    """Return `n` identical report rows."""
    return [
        {
            "report_id":    str(id_offset + i),
            "drug_name":    drug,
            "adverse_event": event,
            "outcome_code": outcome,
            "report_date":  pd.NaT,
            "age_years":    float("nan"),
            "sex":          "UNK",
        }
        for i in range(n)
    ]


def _make_clusterable_df() -> pd.DataFrame:
    """
    Two dense clusters + one isolated noise point.

    Cluster A (drugA/bleeding/HO)  — 10 identical reports
    Cluster B (drugB/myalgia/OT)   — 8 identical reports
    Noise     (uniqueDrug/rareEvent/DE) — 1 isolated report

    Because all reports within each cluster are identical, their cosine
    distance is 0.  Reports from different clusters share no features and
    will have cosine distance = 1.  With eps=0.15 and min_samples=2:
    - Both clusters are found (each has density >> min_samples)
    - The single noise point cannot reach min_samples neighbours → noise
    """
    rows = (
        _make_report("drugA", "bleeding",   "HO", n=10, id_offset=0)
        + _make_report("drugB", "myalgia",  "OT", n=8,  id_offset=10)
        + _make_report("uniqueDrug", "rareEvent", "DE", n=1, id_offset=18)
    )
    return pd.DataFrame(rows)


def _make_single_cluster_df() -> pd.DataFrame:
    """All reports belong to one drug/event combination."""
    rows = _make_report("warfarin", "bleeding", "HO", n=15, id_offset=0)
    return pd.DataFrame(rows)


def _make_no_cluster_df() -> pd.DataFrame:
    """Every report is unique — all will be noise."""
    drugs  = [f"drug{i}" for i in range(10)]
    events = [f"event{i}" for i in range(10)]
    rows = [
        {
            "report_id":    str(i),
            "drug_name":    drugs[i],
            "adverse_event": events[i],
            "outcome_code": "OT",
            "report_date":  pd.NaT,
            "age_years":    float("nan"),
            "sex":          "UNK",
        }
        for i in range(10)
    ]
    return pd.DataFrame(rows)


def _make_multi_drug_cluster_df() -> pd.DataFrame:
    """
    One cluster with reports from two drugs sharing the same event.
    drugX and drugY both report 'headache' / 'OT'.
    With eps=0.15 they will cluster together because they share the
    event and outcome features (partial feature overlap → low cosine distance).
    """
    rows = (
        _make_report("drugX", "headache", "OT", n=8, id_offset=0)
        + _make_report("drugY", "headache", "OT", n=5, id_offset=8)
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:

    def test_returns_ndarray_and_feature_names(self):
        df = _make_clusterable_df()
        matrix, names = build_feature_matrix(df)
        assert isinstance(matrix, np.ndarray)
        assert isinstance(names, list)

    def test_matrix_row_count_matches_dataframe(self):
        df = _make_clusterable_df()
        matrix, _ = build_feature_matrix(df)
        assert matrix.shape[0] == len(df)

    def test_matrix_dtype_is_float32(self):
        df = _make_clusterable_df()
        matrix, _ = build_feature_matrix(df)
        assert matrix.dtype == np.float32

    def test_feature_names_have_correct_prefixes(self):
        df = _make_clusterable_df()
        _, names = build_feature_matrix(df)
        assert any(n.startswith("drug:") for n in names)
        assert any(n.startswith("event:") for n in names)
        assert any(n.startswith("outcome:") for n in names)

    def test_all_values_are_binary(self):
        df = _make_clusterable_df()
        matrix, _ = build_feature_matrix(df)
        unique_vals = set(matrix.flatten().tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_each_row_has_at_least_one_drug_feature_set(self):
        """Every report must have exactly one drug column set to 1."""
        df = _make_clusterable_df()
        matrix, names = build_feature_matrix(df)
        drug_indices = [i for i, n in enumerate(names) if n.startswith("drug:")]
        drug_cols = matrix[:, drug_indices]
        row_sums = drug_cols.sum(axis=1)
        assert (row_sums == 1).all(), "Each row should have exactly one drug feature active"

    def test_empty_dataframe_returns_empty_matrix(self):
        df = pd.DataFrame(columns=["drug_name", "adverse_event", "outcome_code"])
        matrix, names = build_feature_matrix(df)
        assert matrix.shape[0] == 0

    def test_missing_column_raises_value_error(self):
        df = pd.DataFrame({"drug_name": ["aspirin"], "adverse_event": ["nausea"]})
        with pytest.raises(ValueError, match="outcome_code"):
            build_feature_matrix(df)

    def test_identical_reports_have_identical_rows(self):
        """Two reports with the same drug/event/outcome must produce identical rows."""
        df = pd.DataFrame([
            {"drug_name": "aspirin", "adverse_event": "nausea", "outcome_code": "OT"},
            {"drug_name": "aspirin", "adverse_event": "nausea", "outcome_code": "OT"},
            {"drug_name": "aspirin", "adverse_event": "nausea", "outcome_code": "OT"},
        ])
        matrix, _ = build_feature_matrix(df)
        assert np.array_equal(matrix[0], matrix[1])
        assert np.array_equal(matrix[1], matrix[2])

    def test_different_drugs_produce_different_rows(self):
        """Two reports differing only in drug_name must have different feature rows."""
        df = pd.DataFrame([
            {"drug_name": "aspirin",  "adverse_event": "nausea", "outcome_code": "OT"},
            {"drug_name": "warfarin", "adverse_event": "nausea", "outcome_code": "OT"},
        ])
        matrix, _ = build_feature_matrix(df)
        assert not np.array_equal(matrix[0], matrix[1])


# ---------------------------------------------------------------------------
# 2. _top_n and _format_outcomes helpers
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_top_n_returns_most_common_first(self):
        s = pd.Series(["a", "a", "a", "b", "b", "c"])
        assert _top_n(s, 2) == ["a", "b"]

    def test_top_n_respects_limit(self):
        s = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d"] * 1)
        assert len(_top_n(s, 2)) == 2

    def test_top_n_on_single_value(self):
        s = pd.Series(["x"] * 10)
        assert _top_n(s, 3) == ["x"]

    def test_format_outcomes_known_code(self):
        result = _format_outcomes(["HO"])
        assert result == ["HO (Hospitalisation)"]

    def test_format_outcomes_death_code(self):
        result = _format_outcomes(["DE"])
        assert result == ["DE (Death)"]

    def test_format_outcomes_unknown_code_passes_through(self):
        result = _format_outcomes(["ZZ"])
        assert result == ["ZZ (ZZ)"]

    def test_format_outcomes_multiple_codes(self):
        result = _format_outcomes(["HO", "DE"])
        assert len(result) == 2
        assert "HO (Hospitalisation)" in result
        assert "DE (Death)" in result


# ---------------------------------------------------------------------------
# 3. _compute_cluster_stats
# ---------------------------------------------------------------------------

class TestComputeClusterStats:

    @pytest.fixture
    def pure_cluster_df(self) -> pd.DataFrame:
        rows = _make_report("warfarin", "bleeding", "HO", n=10, id_offset=0)
        return pd.DataFrame(rows)

    @pytest.fixture
    def mixed_cluster_df(self) -> pd.DataFrame:
        """7 warfarin + 3 ibuprofen, both reporting bleeding/HO."""
        rows = (
            _make_report("warfarin",  "bleeding", "HO", n=7, id_offset=0)
            + _make_report("ibuprofen", "bleeding", "HO", n=3, id_offset=7)
        )
        return pd.DataFrame(rows)

    def test_size_equals_dataframe_length(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        assert result.size == len(pure_cluster_df)

    def test_dominant_drug_is_most_common(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        assert result.dominant_drug == "warfarin"

    def test_drug_purity_one_for_pure_cluster(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        assert result.drug_purity == pytest.approx(1.0)

    def test_drug_purity_fractional_for_mixed_cluster(self, mixed_cluster_df):
        result = _compute_cluster_stats(mixed_cluster_df, cluster_id=0)
        # 7 warfarin out of 10 = 0.7
        assert result.drug_purity == pytest.approx(0.7)

    def test_dominant_drug_correct_in_mixed_cluster(self, mixed_cluster_df):
        result = _compute_cluster_stats(mixed_cluster_df, cluster_id=0)
        assert result.dominant_drug == "warfarin"

    def test_dominant_events_is_list(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        assert isinstance(result.dominant_events, list)

    def test_dominant_events_not_empty(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        assert len(result.dominant_events) >= 1

    def test_dominant_events_most_common_first(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        assert result.dominant_events[0] == "bleeding"

    def test_dominant_outcomes_formatted_with_label(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        assert any("HO" in o for o in result.dominant_outcomes)
        assert any("Hospitalisation" in o for o in result.dominant_outcomes)

    def test_is_noise_false_for_real_cluster(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        assert result.is_noise is False

    def test_is_noise_true_for_cluster_id_minus_one(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=-1)
        assert result.is_noise is True

    def test_noise_label_contains_noise_marker(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=-1)
        assert "noise" in result.label.lower()

    def test_label_contains_dominant_drug(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        assert "warfarin" in result.label

    def test_label_contains_size(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        assert "n=10" in result.label

    def test_report_ids_populated(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        assert len(result.report_ids) == 10

    def test_event_purity_one_for_single_event(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        assert result.event_purity == pytest.approx(1.0)

    def test_to_dict_contains_all_keys(self, pure_cluster_df):
        result = _compute_cluster_stats(pure_cluster_df, cluster_id=0)
        d = result.to_dict()
        required = {
            "cluster_id", "is_noise", "size", "dominant_drug",
            "dominant_events", "dominant_outcomes",
            "drug_purity", "event_purity", "label",
        }
        assert required.issubset(d.keys())

    def test_to_dict_purity_is_rounded(self, mixed_cluster_df):
        result = _compute_cluster_stats(mixed_cluster_df, cluster_id=0)
        d = result.to_dict()
        # Rounded to 4 decimal places — string representation should not have > 4 dp
        assert len(str(d["drug_purity"]).split(".")[-1]) <= 4


# ---------------------------------------------------------------------------
# 4. cluster_reports — integration tests
# ---------------------------------------------------------------------------

class TestClusterReports:

    # ── Empty input ───────────────────────────────────────────────────────

    def test_empty_dataframe_returns_empty_list(self):
        from core.schemas import REQUIRED_INTERNAL_COLUMNS
        df = pd.DataFrame(columns=REQUIRED_INTERNAL_COLUMNS)
        assert cluster_reports(df) == []

    # ── Return type ───────────────────────────────────────────────────────

    def test_returns_list(self):
        df = _make_clusterable_df()
        results = cluster_reports(df)
        assert isinstance(results, list)

    def test_all_elements_are_cluster_result_instances(self):
        df = _make_clusterable_df()
        results = cluster_reports(df)
        assert all(isinstance(r, ClusterResult) for r in results)

    # ── Two-cluster + noise dataset ───────────────────────────────────────

    def test_finds_two_genuine_clusters(self):
        """
        The clusterable DataFrame has two dense groups (drugA and drugB)
        plus one isolated point.  DBSCAN must find exactly 2 genuine clusters.
        """
        df = _make_clusterable_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        genuine = [r for r in results if not r.is_noise]
        assert len(genuine) == 2, (
            f"Expected 2 genuine clusters, got {len(genuine)}: "
            f"{[r.label for r in results]}"
        )

    def test_noise_cluster_present(self):
        """The single isolated report must be labelled as noise."""
        df = _make_clusterable_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        noise = [r for r in results if r.is_noise]
        assert len(noise) == 1

    def test_noise_cluster_has_size_one(self):
        df = _make_clusterable_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        noise = [r for r in results if r.is_noise][0]
        assert noise.size == 1

    def test_noise_cluster_is_last(self):
        """Noise cluster must always be the last element."""
        df = _make_clusterable_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        assert results[-1].is_noise is True

    def test_cluster_sizes_sum_to_dataframe_length(self):
        """Every report must appear in exactly one cluster (genuine or noise)."""
        df = _make_clusterable_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        total = sum(r.size for r in results)
        assert total == len(df)

    def test_dominant_drug_drugA_found(self):
        """One genuine cluster must be dominated by drugA."""
        df = _make_clusterable_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        genuine = [r for r in results if not r.is_noise]
        dominant_drugs = {r.dominant_drug for r in genuine}
        assert "drugA" in dominant_drugs

    def test_dominant_drug_drugB_found(self):
        """The other genuine cluster must be dominated by drugB."""
        df = _make_clusterable_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        genuine = [r for r in results if not r.is_noise]
        dominant_drugs = {r.dominant_drug for r in genuine}
        assert "drugB" in dominant_drugs

    def test_drug_purity_one_for_pure_clusters(self):
        """Each cluster in the two-cluster dataset has purity = 1.0."""
        df = _make_clusterable_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        genuine = [r for r in results if not r.is_noise]
        for r in genuine:
            assert r.drug_purity == pytest.approx(1.0), (
                f"Cluster '{r.label}' has drug_purity={r.drug_purity}, expected 1.0"
            )

    def test_genuine_clusters_sorted_by_size_descending(self):
        """Genuine clusters (before the noise entry) are sorted largest-first."""
        df = _make_clusterable_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        genuine = [r for r in results if not r.is_noise]
        sizes = [r.size for r in genuine]
        assert sizes == sorted(sizes, reverse=True)

    # ── Single-cluster dataset ────────────────────────────────────────────

    def test_single_cluster_no_noise(self):
        """When all reports are identical, there is one cluster and no noise."""
        df = _make_single_cluster_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        genuine = [r for r in results if not r.is_noise]
        noise = [r for r in results if r.is_noise]
        assert len(genuine) == 1
        assert len(noise) == 0

    def test_single_cluster_dominant_drug(self):
        df = _make_single_cluster_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        assert results[0].dominant_drug == "warfarin"

    def test_single_cluster_dominant_event(self):
        df = _make_single_cluster_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        assert "bleeding" in results[0].dominant_events

    def test_single_cluster_size_matches(self):
        df = _make_single_cluster_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        assert results[0].size == len(df)

    # ── All-noise dataset ─────────────────────────────────────────────────

    def test_all_noise_produces_one_noise_cluster(self):
        """When every report is unique, DBSCAN produces only noise."""
        df = _make_no_cluster_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        genuine = [r for r in results if not r.is_noise]
        noise   = [r for r in results if r.is_noise]
        assert len(genuine) == 0
        assert len(noise) == 1

    def test_noise_cluster_covers_all_reports_when_all_unique(self):
        df = _make_no_cluster_df()
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=2)
        assert results[0].size == len(df)

    # ── Parameter sensitivity ─────────────────────────────────────────────

    def test_high_eps_merges_clusters(self):
        """
        With eps=1.0 (maximum cosine distance) all points are reachable from
        all others → one giant cluster, no noise.
        """
        df = _make_clusterable_df()
        results = cluster_reports(df, eps=1.0, min_samples=2)
        genuine = [r for r in results if not r.is_noise]
        # At eps=1.0 everything merges into one cluster
        assert len(genuine) >= 1
        total = sum(r.size for r in genuine)
        # Most (or all) reports should be in genuine clusters, not noise
        assert total >= len(df) - 1  # allow 1 noise point at most

    def test_high_min_samples_increases_noise(self):
        """
        Setting min_samples higher than the cluster size forces all reports
        into noise.
        """
        df = _make_single_cluster_df()   # 15 identical reports
        # min_samples=20 > 15 → even the dense group can't form a core point
        results = cluster_reports(df, eps=DEFAULT_EPS, min_samples=20)
        genuine = [r for r in results if not r.is_noise]
        assert len(genuine) == 0

    # ── to_dict serialisation ─────────────────────────────────────────────

    def test_to_dict_works_on_all_results(self):
        df = _make_clusterable_df()
        results = cluster_reports(df)
        for r in results:
            d = r.to_dict()
            assert isinstance(d, dict)
            assert "cluster_id" in d
            assert "label" in d

    # ── Sample dataset integration ────────────────────────────────────────

    def test_cluster_reports_on_sample_csv(self, clean_df):
        """
        Smoke test: run clustering on the bundled sample dataset.
        We do not assert specific cluster assignments (they depend on the
        exact CSV contents), but we do assert structural invariants.
        """
        results = cluster_reports(clean_df)
        assert isinstance(results, list)

        # All sizes must be positive
        assert all(r.size > 0 for r in results)

        # Total reports across all clusters == DataFrame length
        assert sum(r.size for r in results) == len(clean_df)

        # Noise cluster, if present, must be last
        if any(r.is_noise for r in results):
            assert results[-1].is_noise

        # Every result must be serialisable
        for r in results:
            d = r.to_dict()
            assert d["dominant_drug"] != ""
