"""Branch and identity coverage for fx1 operation audits and features.

Synthetic rows only. Assertions are fail-closed errors or exact numeric results.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fx1.operations import audit_bar_integrity as bars
from fx1.operations import audit_cross_field_contracts as contracts
from fx1.operations import audit_missingness as missingness
from fx1.operations import audit_missingness_association as association
from fx1.operations import audit_monotonic_sequences as monotonic
from fx1.operations import audit_referential_integrity as referential
from fx1.operations import audit_revision_conflicts as revisions
from fx1.operations import audit_schema_drift as drift
from fx1.operations import audit_source_coverage as coverage
from fx1.operations import bipower_variation as bipower
from fx1.operations import join_asof_observations as join_asof
from fx1.operations import permutation_entropy as entropy
from fx1.operations import resolve_security_identity as identity
from fx1.operations import rolling_autocorrelation as autocorr
from fx1.operations import rolling_mad as mad
from fx1.operations import rolling_rank as rank
from fx1.operations import select_asof_revisions as asof
from fx1.operations import select_universe_membership as universe
from fx1.operations import spectral_summary as spectral
from fx1.operations import summarize_ingestion_latency as latency
from fx1.operations import time_weighted_mean as twmean
from fx1.operations.base import OperationContext

OVERFLOW_CLOCK = "9999-12-31T23:59:59.999999-23:59"
NAIVE_CLOCK = "2026-01-01T00:00:00"
LONG_CLOCK = "2026-01-01T00:00:00+00:00" + ("0" * 40)


class _NullOffset(tzinfo):
    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return None

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None

    def tzname(self, dt: datetime | None) -> str:
        return "NULL"


def _rejects(model: Any, payload: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        model.model_validate(payload)


def _run(module: Any, payload: dict[str, Any], context: OperationContext) -> Any:
    return module.execute(module.Input.model_validate(payload), context)


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def _wide_rows(count: int, width: int = 64) -> list[dict[str, int]]:
    row = {f"f{index:02d}": 0 for index in range(width)}
    return [row] * count


# ---------------------------------------------------------------------------
# audit_cross_field_contracts
# ---------------------------------------------------------------------------


def test_cross_field_kind_and_operator_matrix() -> None:
    assert contracts._kind(None, compatible=True) == "null"
    assert contracts._kind(True, compatible=False) == "boolean"
    assert contracts._kind(1, compatible=True) == "number"
    assert contracts._kind(1, compatible=False) == "integer"
    assert contracts._kind(1.5, compatible=True) == "number"
    assert contracts._kind(1.5, compatible=False) == "float"
    assert contracts._kind("a", compatible=True) == "string"
    assert contracts._compare(2, 3, "lt", compatible=True) is True
    assert contracts._compare(2, 2, "le", compatible=True) is True
    assert contracts._compare(5, 4, "gt", compatible=True) is True
    assert contracts._compare(5, 5, "ge", compatible=True) is True
    assert contracts._compare(1, 1, "eq", compatible=True) is True
    assert contracts._compare(1, 2, "ne", compatible=True) is True
    assert contracts._compare(True, 1, "lt", compatible=True) is None
    assert contracts._compare(1, True, "gt", compatible=True) is None
    assert contracts._compare("a", "b", "lt", compatible=True) is None
    assert contracts._compare(1, "b", "le", compatible=True) is None
    assert contracts._compare(1, 1.0, "lt", compatible=False) is None
    with pytest.raises(ValueError, match="unsupported comparison"):
        contracts._compare(1, 2, "xx", compatible=True)


def test_cross_field_numeric_identity_compatible_and_distinct(context: OperationContext) -> None:
    rows = [{"x": 1, "y": 1.0}, {"x": 2, "y": 2}, {"x": 3, "y": 1}]
    rules = [
        {"id": "le", "left_field": "x", "operator": "le", "right": {"kind": "field", "field": "y"}},
        {"id": "gt", "left_field": "x", "operator": "gt", "right": {"kind": "literal", "value": 1}},
        {"id": "ge", "left_field": "x", "operator": "ge", "right": {"kind": "literal", "value": 3}},
        {"id": "eq", "left_field": "x", "operator": "eq", "right": {"kind": "field", "field": "y"}},
        {"id": "ne", "left_field": "x", "operator": "ne", "right": {"kind": "literal", "value": 2}},
    ]
    result = _run(contracts, {"rows": rows, "rules": rules}, context)
    assert result.checked_comparisons == 15
    assert result.failed_comparisons == 6
    assert result.conforming_rows == 0
    assert result.nonconforming_rows == 3
    assert result.unassessed_rows == 0
    assert result.passed is False
    by_rule = {item.rule_id: item for item in result.rules}
    assert by_rule["le"].passed_comparisons == 2
    assert by_rule["le"].failure_counts == {"comparison_failed": 1}
    assert by_rule["gt"].passed_comparisons == 2
    assert by_rule["eq"].passed_comparisons == 2
    assert by_rule["ne"].failed_comparisons == 1
    assert by_rule["ge"].passed_comparisons == 1
    distinct = _run(
        contracts, {"rows": rows, "rules": rules, "numeric_policy": "distinct"}, context
    )
    distinct_rules = {item.rule_id: item for item in distinct.rules}
    assert distinct_rules["le"].failure_counts["incompatible_types"] == 1
    assert distinct_rules["eq"].failure_counts["comparison_failed"] == 2
    assert distinct.numeric_policy == "distinct"


def test_cross_field_missing_null_and_type_policies(context: OperationContext) -> None:
    payload = {
        "rows": [
            {"a": 1, "b": 1, "flag": True, "label": "b", "hole": None},
            {"a": 1},
            {},
            {"a": 1, "b": 1, "flag": False, "label": "a", "hole": 0},
        ],
        "rules": [
            {
                "id": "skip_a",
                "left_field": "a",
                "operator": "eq",
                "right": {"kind": "literal", "value": 1},
                "missing_policy": "skip",
            },
            {
                "id": "need_b",
                "left_field": "b",
                "operator": "eq",
                "right": {"kind": "field", "field": "a"},
            },
            {
                "id": "same_field",
                "left_field": "a",
                "operator": "eq",
                "right": {"kind": "field", "field": "a"},
            },
            {
                "id": "null_skip",
                "left_field": "hole",
                "operator": "eq",
                "right": {"kind": "literal", "value": 0},
                "null_policy": "skip",
            },
            {
                "id": "null_fail",
                "left_field": "hole",
                "operator": "eq",
                "right": {"kind": "literal", "value": 0},
                "null_policy": "fail",
            },
            {
                "id": "null_eq",
                "left_field": "hole",
                "operator": "eq",
                "right": {"kind": "literal", "value": None},
                "null_policy": "compare",
            },
            {
                "id": "null_lt",
                "left_field": "hole",
                "operator": "lt",
                "right": {"kind": "literal", "value": 1},
                "null_policy": "compare",
            },
            {
                "id": "bool_lt",
                "left_field": "flag",
                "operator": "lt",
                "right": {"kind": "literal", "value": True},
            },
            {
                "id": "bool_eq_int",
                "left_field": "flag",
                "operator": "eq",
                "right": {"kind": "literal", "value": 1},
            },
            {
                "id": "text_gt",
                "left_field": "label",
                "operator": "gt",
                "right": {"kind": "literal", "value": "a"},
            },
            {
                "id": "text_eq",
                "left_field": "label",
                "operator": "eq",
                "right": {"kind": "literal", "value": "b"},
            },
        ],
    }
    result = _run(contracts, payload, context)
    by_rule = {item.rule_id: item for item in result.rules}
    assert by_rule["skip_a"].skipped_missing == 1
    assert by_rule["skip_a"].passed_comparisons == 3
    assert by_rule["skip_a"].passed is True
    need_b = [item for item in result.diagnostics if item.rule_id == "need_b"]
    assert [item.missing_fields for item in need_b] == [["b"], ["b", "a"]]
    same = [item for item in result.diagnostics if item.rule_id == "same_field"]
    assert same[0].missing_fields == ["a"]
    assert by_rule["null_skip"].skipped_null == 1
    assert by_rule["null_fail"].failure_counts["null_operand"] == 1
    assert by_rule["null_eq"].passed_comparisons == 1
    assert by_rule["null_lt"].failure_counts["incompatible_types"] == 1
    assert by_rule["bool_lt"].failure_counts == {"incompatible_types": 2, "missing_field": 2}
    assert by_rule["bool_eq_int"].failure_counts["comparison_failed"] == 2
    assert by_rule["text_gt"].failure_counts["incompatible_types"] == 2
    assert by_rule["text_eq"].passed_comparisons == 1
    assert result.rules[0].passed is True
    capped = _run(contracts, {**payload, "max_diagnostics": 1}, context)
    assert len(capped.diagnostics) == 1
    assert capped.omitted_diagnostics == capped.failed_comparisons - 1
    silent = _run(contracts, {**payload, "max_diagnostics": 0}, context)
    assert silent.diagnostics == []
    assert silent.omitted_diagnostics == silent.failed_comparisons


def test_cross_field_assessments_and_bounds(context: OperationContext) -> None:
    empty = _run(
        contracts,
        {
            "rows": [],
            "rules": [
                {
                    "id": "only",
                    "left_field": "a",
                    "operator": "eq",
                    "right": {"kind": "literal", "value": 1},
                }
            ],
        },
        context,
    )
    assert empty.assessment == "no_assessed_comparisons"
    assert empty.passed is None
    assert empty.rules[0].passed is None
    skipped = _run(
        contracts,
        {
            "rows": [{}],
            "rules": [
                {
                    "id": "skip",
                    "left_field": "a",
                    "operator": "eq",
                    "right": {"kind": "literal", "value": 1},
                    "missing_policy": "skip",
                }
            ],
        },
        context,
    )
    assert skipped.assessment == "no_assessed_comparisons"
    assert skipped.skipped_comparisons == 1
    assert skipped.unassessed_rows == 1
    assert skipped.rules[0].passed is None
    passed = _run(
        contracts,
        {
            "rows": [{"a": 2, "b": 4}],
            "rules": [
                {
                    "id": "ok",
                    "left_field": "b",
                    "operator": "ge",
                    "right": {"kind": "field", "field": "a"},
                }
            ],
        },
        context,
    )
    assert passed.passed is True
    assert passed.assessment == "passed"
    assert passed.conforming_rows == 1
    assert passed.rules[0].passed is True
    rule = {
        "id": "r",
        "left_field": "a",
        "operator": "eq",
        "right": {"kind": "literal", "value": 1},
    }
    _rejects(
        contracts.Input,
        {"rules": [rule, {**rule, "id": "r"}]},
        "unique",
    )
    _rejects(
        contracts.Input,
        {"rows": [{}] * 3907, "rules": [{**rule, "id": f"r{i}"} for i in range(64)]},
        "250000 row/rule",
    )
    _rejects(
        contracts.Input, {"rows": [{f"f{i}": 1 for i in range(65)}], "rules": [rule]}, "64 fields"
    )
    _rejects(contracts.Input, {"rows": _wide_rows(3907), "rules": [rule]}, "250000 fields")


# ---------------------------------------------------------------------------
# audit_referential_integrity
# ---------------------------------------------------------------------------


def test_referential_token_kinds() -> None:
    assert referential._token(True, compatible_numbers=True) == ("boolean", True)
    assert referential._token(1, compatible_numbers=True) == ("number", 1)
    assert referential._token(1, compatible_numbers=False) == ("integer", 1)
    assert referential._token(1.5, compatible_numbers=True) == ("number", 1.5)
    assert referential._token(1.5, compatible_numbers=False) == ("float", 1.5)
    assert referential._token(None, compatible_numbers=False) == ("null", None)
    assert referential._token("k", compatible_numbers=True) == ("string", "k")


def test_referential_matches_duplicates_and_null_policies(context: OperationContext) -> None:
    parents = [
        {"id": 1, "venue": "X"},
        {"id": 1.0, "venue": "X"},
        {"id": True, "venue": "X"},
        {"id": None, "venue": "X"},
        {"venue": "X"},
        {"id": 2, "venue": None},
        {"id": "ok", "venue": "Y"},
    ]
    children = [
        {"child_id": 1, "child_venue": "X"},
        {"child_id": 1.0, "child_venue": "X"},
        {"child_id": True, "child_venue": "X"},
        {"child_id": "missing", "child_venue": "Z"},
        {"child_venue": "Y"},
        {"child_id": None, "child_venue": "Y"},
        {"child_id": "ok", "child_venue": "Y"},
    ]
    result = _run(
        referential,
        {
            "parent_rows": parents,
            "child_rows": children,
            "parent_keys": ["id", "venue"],
            "child_keys": ["child_id", "child_venue"],
            "numeric_policy": "distinct",
            "child_null_policy": "reject",
            "max_parent_indices": 1,
        },
        context,
    )
    assert result.passed is False
    assert result.invalid_parent_rows == 3
    assert result.parent_rows_with_missing_fields == 1
    assert result.parent_rows_with_null_fields == 2
    assert result.valid_parent_rows == 4
    assert result.duplicate_parent_key_groups == 0
    assert result.matched_child_rows == 4
    assert result.unmatched_child_rows == 1
    assert result.invalid_child_rows == 2
    assert result.ignored_null_child_rows == 0
    codes = [item.code for item in result.diagnostics]
    assert "invalid_parent_key" in codes
    assert "unmatched_child_reference" in codes
    assert "invalid_child_key" in codes
    compatible = _run(
        referential,
        {
            "parent_rows": [{"id": 1}, {"id": 1}, {"id": 1}],
            "child_rows": [{"id": 1.0}, {"id": None}, {"id": 9}],
            "parent_keys": ["id"],
            "numeric_policy": "compatible",
            "child_null_policy": "ignore",
            "max_diagnostics": 1,
            "max_parent_indices": 1,
        },
        context,
    )
    assert compatible.duplicate_parent_key_groups == 1
    assert compatible.extra_duplicate_parent_rows == 2
    assert compatible.ambiguous_child_rows == 1
    assert compatible.ignored_null_child_rows == 1
    assert compatible.unmatched_child_rows == 1
    assert compatible.diagnostics[0].omitted_parent_indices == 2
    assert len(compatible.diagnostics[0].parent_row_indices) == 1
    assert compatible.omitted_diagnostics == compatible.diagnostic_count - 1
    assert compatible.child_keys == ["id"]
    assert compatible.assessment == "failed"


def test_referential_empty_and_clean_assessments(context: OperationContext) -> None:
    clean = _run(
        referential,
        {
            "parent_rows": [{"id": "A", "extra": 1}],
            "child_rows": [{"id": "A"}],
            "parent_keys": ["id"],
        },
        context,
    )
    assert clean.passed is True
    assert clean.assessment == "passed"
    assert clean.matched_child_rows == 1
    assert clean.distinct_parent_keys == 1
    no_children = _run(
        referential,
        {"parent_rows": [{"id": "A"}], "child_rows": [], "parent_keys": ["id"]},
        context,
    )
    assert no_children.passed is None
    assert no_children.assessment == "no_comparable_references"
    empty = _run(
        referential,
        {"parent_rows": [], "child_rows": [], "parent_keys": ["id"]},
        context,
    )
    assert empty.passed is None
    assert empty.duplicate_parent_key_groups == 0
    parents_only_bad = _run(
        referential,
        {"parent_rows": [{"other": 1}], "child_rows": [], "parent_keys": ["id"]},
        context,
    )
    assert parents_only_bad.passed is False
    assert parents_only_bad.comparable_child_rows == 0


def test_referential_bounds(context: OperationContext) -> None:
    _rejects(
        referential.Input,
        {"parent_keys": ["a", "a"], "child_rows": [], "parent_rows": []},
        "distinct",
    )
    _rejects(
        referential.Input,
        {"parent_keys": ["a", "b"], "child_keys": ["a"], "parent_rows": [], "child_rows": []},
        "equal lengths",
    )
    _rejects(
        referential.Input,
        {
            "parent_keys": ["a", "b"],
            "child_keys": ["a", "a"],
            "parent_rows": [],
            "child_rows": [],
        },
        "child_keys must be distinct",
    )
    _rejects(
        referential.Input,
        {"parent_keys": ["a"], "parent_rows": [{f"f{i}": 1 for i in range(65)}], "child_rows": []},
        "64 fields",
    )
    _rejects(
        referential.Input,
        {"parent_keys": ["f00"], "parent_rows": _wide_rows(3907), "child_rows": []},
        "250000 fields",
    )


# ---------------------------------------------------------------------------
# select_universe_membership
# ---------------------------------------------------------------------------


def _member(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "membership_id": "m1",
        "security_id": "AAA",
        "state": "included",
        "effective_from": "2026-01-01T00:00:00Z",
        "effective_to": None,
        "available_time": "2026-01-01T00:00:00Z",
        "revision_id": "r1",
        "source": "src",
    }
    row.update(updates)
    return row


def test_universe_inclusion_exclusion_and_expiry(context: OperationContext) -> None:
    rows = [
        _member(),
        _member(
            membership_id="m2",
            state="excluded",
            effective_from="2026-02-01T00:00:00Z",
            effective_to="2026-03-01T00:00:00Z",
            available_time="2026-01-15T00:00:00Z",
        ),
    ]
    included = _run(
        universe,
        {"memberships": rows, "decision_time": "2026-01-15T00:00:00Z"},
        context,
    )
    assert included.included_count == 1
    assert included.selections[0].status == "included"
    assert included.selections[0].selected_row_index == 0
    excluded = _run(
        universe,
        {"memberships": rows, "decision_time": "2026-02-15T00:00:00Z"},
        context,
    )
    assert excluded.selections[0].status == "excluded"
    assert excluded.selections[0].selected_row_index == 1
    revealed = _run(
        universe,
        {"memberships": rows, "decision_time": "2026-03-01T00:00:00Z"},
        context,
    )
    assert revealed.selections[0].status == "included"
    assert revealed.selections[0].selected_row_index == 0
    boundary = _run(
        universe,
        {"memberships": rows, "decision_time": "2026-01-01T00:00:00Z"},
        context,
    )
    assert boundary.selections[0].status == "included"
    before = _run(
        universe,
        {
            "memberships": [_member(effective_from="2026-06-01T00:00:00Z")],
            "decision_time": "2026-01-01T00:00:00Z",
        },
        context,
    )
    assert before.selections[0].status == "missing"


def test_universe_revisions_conflicts_and_ties(context: OperationContext) -> None:
    replaced = _run(
        universe,
        {
            "memberships": [
                _member(state="included", available_time="2026-01-01T00:00:00Z", revision_id="r1"),
                _member(state="excluded", available_time="2026-02-01T00:00:00Z", revision_id="r2"),
                _member(state="included", available_time="2025-01-01T00:00:00Z", revision_id="r9"),
            ],
            "decision_time": "2026-03-01T00:00:00Z",
        },
        context,
    )
    assert replaced.selections[0].status == "excluded"
    assert replaced.selections[0].selected_row_index == 1
    assert replaced.unavailable_input_rows == 0
    conflict = _run(
        universe,
        {
            "memberships": [
                _member(source="left", available_time="2026-01-01T00:00:00Z"),
                _member(source="right", available_time="2026-01-01T00:00:00Z"),
            ],
            "decision_time": "2026-02-01T00:00:00Z",
        },
        context,
    )
    assert conflict.selections[0].status == "ambiguous"
    assert conflict.selections[0].selected_row_index is None
    assert conflict.selections[0].conflicting_membership_ids == 1
    assert conflict.ambiguous_count == 1
    states = _run(
        universe,
        {
            "memberships": [
                _member(membership_id="keep", state="included"),
                _member(membership_id="drop", state="excluded"),
            ],
            "decision_time": "2026-02-01T00:00:00Z",
        },
        context,
    )
    assert states.selections[0].status == "ambiguous"
    assert states.selections[0].conflicting_membership_ids == 0
    tie = _run(
        universe,
        {
            "memberships": [
                _member(
                    membership_id="m-late", revision_id="a", available_time="2026-01-02T00:00:00Z"
                ),
                _member(
                    membership_id="m-early", revision_id="z", available_time="2026-01-02T00:00:00Z"
                ),
                _member(
                    membership_id="m-old", revision_id="z", available_time="2026-01-01T00:00:00Z"
                ),
            ],
            "decision_time": "2026-02-01T00:00:00Z",
        },
        context,
    )
    assert tie.selections[0].status == "included"
    assert tie.selections[0].selected_row_index == 1
    same_revision = _run(
        universe,
        {
            "memberships": [
                _member(revision_id="m", available_time="2026-01-01T00:00:00Z"),
                _member(revision_id="m", available_time="2026-01-01T00:00:00Z"),
            ],
            "decision_time": "2026-02-01T00:00:00Z",
        },
        context,
    )
    assert same_revision.selections[0].selected_row_index == 0
    future = _run(
        universe,
        {
            "memberships": [_member(available_time="2026-06-01T00:00:00Z")],
            "decision_time": "2026-01-01T00:00:00Z",
        },
        context,
    )
    assert future.unavailable_input_rows == 1
    assert future.selections[0].status == "missing"
    assert future.missing_count == 1


def test_universe_pagination_lineage_and_clocks(context: OperationContext) -> None:
    rows = [_member(security_id=name, membership_id=name) for name in ("CCC", "AAA", "BBB", "DDD")]
    page = _run(
        universe,
        {
            "memberships": rows,
            "decision_time": "2026-02-01T00:00:00Z",
            "offset": 1,
            "limit": 1,
            "security_ids": ["AAA", "BBB", "CCC"],
        },
        context,
    )
    assert [item.security_id for item in page.selections] == ["BBB"]
    assert page.security_count == 3
    assert page.included_count == 3
    assert page.next_offset == 2
    assert page.offset == 1
    assert len(page.membership_lineage) == 1
    crowded = _run(
        universe,
        {
            "memberships": [
                _member(membership_id=f"m{index}", revision_id=f"r{index}") for index in range(4)
            ],
            "decision_time": "2026-02-01T00:00:00Z",
            "max_lineage_per_security": 2,
        },
        context,
    )
    assert crowded.selections[0].omitted_supporting_rows == 2
    assert len(crowded.selections[0].supporting_row_indices) == 2
    explicit_missing = _run(
        universe,
        {
            "memberships": rows,
            "decision_time": "2026-02-01T00:00:00Z",
            "security_ids": ["ZZZ"],
        },
        context,
    )
    assert explicit_missing.selections[0].status == "missing"
    empty = _run(universe, {"memberships": [], "decision_time": "2026-02-01T00:00:00Z"}, context)
    assert empty.security_count == 0
    assert empty.next_offset is None
    assert empty.selections == []
    aware = datetime(2026, 2, 1, tzinfo=UTC)
    via_datetime = _run(
        universe,
        {"memberships": [_member()], "decision_time": aware},
        context,
    )
    assert via_datetime.selections[0].status == "included"
    offset_clock = _run(
        universe,
        {
            "memberships": [_member(effective_from="2026-01-01T05:00:00+05:00")],
            "decision_time": "2026-01-01T00:00:00Z",
        },
        context,
    )
    assert offset_clock.selections[0].status == "included"
    past_end = _run(
        universe,
        {
            "memberships": [
                _member(security_id="AAA"),
                _member(security_id="BBB", membership_id="m2"),
            ],
            "decision_time": "2026-02-01T00:00:00Z",
            "offset": 5,
        },
        context,
    )
    assert past_end.selections == []
    assert past_end.next_offset is None
    assert past_end.included_count == 2


def test_universe_clock_and_interval_failures() -> None:
    _rejects(
        universe.Input,
        {"memberships": [_member()], "decision_time": NAIVE_CLOCK},
        "timezone",
    )
    _rejects(
        universe.Input,
        {"memberships": [_member()], "decision_time": 1_700_000_000},
        "timezone-aware",
    )
    _rejects(
        universe.Input,
        {"memberships": [_member()], "decision_time": OVERFLOW_CLOCK},
        "UTC",
    )
    _rejects(
        universe.Input,
        {"memberships": [_member()], "decision_time": datetime(2026, 1, 1, tzinfo=_NullOffset())},
        "timezone",
    )
    _rejects(
        universe.Input,
        {
            "memberships": [_member(effective_to="2026-01-01T00:00:00Z")],
            "decision_time": "2026-02-01T00:00:00Z",
        },
        "strictly after",
    )
    _rejects(
        universe.Input,
        {
            "memberships": [],
            "decision_time": "2026-02-01T00:00:00Z",
            "security_ids": ["AAA", "AAA"],
        },
        "distinct",
    )
    _rejects(
        universe.Input,
        {
            "memberships": [_member(effective_from=NAIVE_CLOCK)],
            "decision_time": "2026-02-01T00:00:00Z",
        },
        "timezone",
    )
    _rejects(
        universe.Input,
        {
            "memberships": [_member(effective_to="not-a-clock")],
            "decision_time": "2026-02-01T00:00:00Z",
        },
        "timezone-aware|clock",
    )


# ---------------------------------------------------------------------------
# audit_schema_drift
# ---------------------------------------------------------------------------


def test_schema_drift_changes_and_policies(context: OperationContext) -> None:
    reference = [{"a": 1, "gone": True, "same": "x"}, {"a": 2, "gone": False, "same": "y"}]
    candidate = [{"a": None, "new": 1.5, "same": "x"}, {"new": 2.5, "same": "y"}]
    result = _run(
        drift,
        {"reference_rows": reference, "candidate_rows": candidate, "max_changes": 1},
        context,
    )
    assert result.assessment == "compared"
    assert result.schema_equal is False
    assert result.changed_field_count == 3
    assert result.omitted_changed_fields == 2
    assert [item.name for item in result.changes] == ["a"]
    assert result.changes[0].changes == [
        "non_null_types_changed",
        "optionality_changed",
        "nullability_changed",
    ]
    assert result.change_counts["added_field"] == 1
    assert result.change_counts["removed_field"] == 1
    full = _run(
        drift,
        {"reference_rows": reference, "candidate_rows": candidate, "max_changes": 10},
        context,
    )
    names = {item.name: item.changes for item in full.changes}
    assert names["gone"] == ["removed_field"]
    assert names["new"] == ["added_field"]
    equal = _run(
        drift,
        {
            "reference_rows": [{"n": 1, "f": 1.0, "b": True, "s": "a", "z": None}],
            "candidate_rows": [{"n": 2, "f": 2.0, "b": False, "s": "b", "z": None}],
            "numeric_policy": "distinct",
        },
        context,
    )
    assert equal.schema_equal is True
    assert equal.changes == []
    profiles = {item.name: item.non_null_types for item in equal.reference_fields}
    assert profiles == {
        "b": ["boolean"],
        "f": ["float"],
        "n": ["integer"],
        "s": ["string"],
        "z": [],
    }
    compatible = _run(
        drift,
        {
            "reference_rows": [{"n": 1}],
            "candidate_rows": [{"n": 1.0}],
            "numeric_policy": "compatible",
        },
        context,
    )
    assert compatible.schema_equal is True
    assert compatible.reference_fields[0].non_null_types == ["number"]
    distinct = _run(
        drift,
        {
            "reference_rows": [{"n": 1}],
            "candidate_rows": [{"n": 1.0}],
            "numeric_policy": "distinct",
        },
        context,
    )
    assert distinct.changes[0].changes == ["non_null_types_changed"]


def test_schema_drift_empty_tables_and_bounds(context: OperationContext) -> None:
    insufficient = _run(drift, {"reference_rows": [], "candidate_rows": [{"a": 1}]}, context)
    assert insufficient.assessment == "insufficient_observations"
    assert insufficient.schema_equal is None
    assert insufficient.candidate_fields[0].name == "a"
    both_empty = _run(drift, {"reference_rows": [], "candidate_rows": []}, context)
    assert both_empty.assessment == "insufficient_observations"
    empty_dicts = _run(drift, {"reference_rows": [{}], "candidate_rows": [{}]}, context)
    assert empty_dicts.assessment == "compared"
    assert empty_dicts.schema_equal is True
    assert empty_dicts.changed_field_count == 0
    optional = _run(
        drift,
        {"reference_rows": [{"a": 1}, {"a": 1}], "candidate_rows": [{"a": 1}, {}]},
        context,
    )
    assert optional.changes[0].changes == ["optionality_changed"]
    assert optional.reference_fields[0].optional is False
    assert optional.candidate_fields[0].optional is True
    _rejects(drift.Input, {"reference_rows": [{f"f{i}": 1 for i in range(65)}]}, "64 fields")
    wide_names = []
    for block in range(5):
        wide_names.append({f"n{block}_{index}": 1 for index in range(60)})
    _rejects(drift.Input, {"reference_rows": wide_names}, "256 distinct")
    _rejects(drift.Input, {"reference_rows": _wide_rows(3907)}, "250000 fields")


# ---------------------------------------------------------------------------
# audit_source_coverage
# ---------------------------------------------------------------------------


def _observation(source: str, security: str, event: str, available: str) -> dict[str, str]:
    return {
        "source": source,
        "security_id": security,
        "event_time": event,
        "available_time": available,
    }


def test_source_coverage_selection_staleness_and_unexpected(context: OperationContext) -> None:
    observations = [
        _observation("s", "AAA", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        _observation("s", "AAA", "2026-01-01T08:00:00Z", "2026-01-01T07:00:00Z"),
        _observation("s", "AAA", "2026-01-01T08:00:00Z", "2026-01-01T09:00:00Z"),
        _observation("s", "AAA", "2026-01-01T11:00:00Z", "2026-01-01T09:00:00Z"),
        _observation("s", "AAA", "2026-01-01T09:00:00Z", "2026-01-01T12:00:00Z"),
        _observation("s", "CCC", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        _observation("s", "DDD", "2026-01-01T08:00:00Z", "2026-01-01T08:00:00Z"),
        _observation("s", "EEE", "2026-01-01T08:00:00Z", "2026-01-01T12:00:00Z"),
    ]
    result = _run(
        coverage,
        {
            "expected_pairs": [
                {"source": "s", "security_id": "AAA"},
                {"source": "s", "security_id": "BBB"},
                {"source": "s", "security_id": "CCC"},
            ],
            "observations": observations,
            "decision_time": "2026-01-01T10:00:00Z",
            "max_age_seconds": 3 * 3600,
            "staleness_clock": "event_time",
        },
        context,
    )
    by_security = {item.security_id: item for item in result.expected_pairs}
    assert by_security["AAA"].status == "covered"
    assert by_security["AAA"].selected_row_index == 2
    assert by_security["AAA"].event_age_seconds == pytest.approx(2 * 3600)
    assert by_security["BBB"].status == "missing"
    assert by_security["BBB"].selected_row_index is None
    assert by_security["CCC"].status == "stale"
    assert result.covered_expected_pairs == 1
    assert result.missing_expected_pairs == 1
    assert result.stale_expected_pairs == 1
    assert result.assessment == "incomplete"
    assert result.complete_expected_coverage is False
    assert result.unavailable_records == 2
    assert result.future_event_records_excluded == 1
    assert result.unexpected_eligible_pair_count == 1
    assert result.unexpected_eligible_pairs[0].security_id == "DDD"
    assert result.unexpected_input_pair_count == 2
    fresh_clock = _run(
        coverage,
        {
            "expected_pairs": [{"source": "s", "security_id": "AAA"}],
            "observations": [
                _observation("s", "AAA", "2026-01-01T09:00:00Z", "2026-01-01T00:00:00Z")
            ],
            "decision_time": "2026-01-01T10:00:00Z",
            "max_age_seconds": 2 * 3600,
            "staleness_clock": "available_time",
        },
        context,
    )
    assert fresh_clock.expected_pairs[0].status == "stale"
    event_clock = _run(
        coverage,
        {
            "expected_pairs": [{"source": "s", "security_id": "AAA"}],
            "observations": [
                _observation("s", "AAA", "2026-01-01T09:00:00Z", "2026-01-01T00:00:00Z")
            ],
            "decision_time": "2026-01-01T10:00:00Z",
            "max_age_seconds": 2 * 3600,
            "staleness_clock": "event_time",
        },
        context,
    )
    assert event_clock.expected_pairs[0].status == "covered"
    exact_age = _run(
        coverage,
        {
            "expected_pairs": [{"source": "s", "security_id": "AAA"}],
            "observations": [
                _observation("s", "AAA", "2026-01-01T08:00:00Z", "2026-01-01T08:00:00Z")
            ],
            "decision_time": "2026-01-01T10:00:00Z",
            "max_age_seconds": 2 * 3600,
        },
        context,
    )
    assert exact_age.expected_pairs[0].status == "covered"
    no_limit = _run(
        coverage,
        {
            "expected_pairs": [{"source": "s", "security_id": "AAA"}],
            "observations": [
                _observation("s", "AAA", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
            ],
            "decision_time": "2026-01-01T10:00:00Z",
        },
        context,
    )
    assert no_limit.assessment == "covered"
    assert no_limit.complete_expected_coverage is True


def test_source_coverage_paging_and_clock_failures(context: OperationContext) -> None:
    paged = _run(
        coverage,
        {
            "expected_pairs": [
                {"source": "s", "security_id": "AAA"},
                {"source": "s", "security_id": "BBB"},
                {"source": "s", "security_id": "CCC"},
            ],
            "observations": [
                _observation("s", "ZZZ", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
                _observation("t", "YYY", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            ],
            "decision_time": "2026-01-01T10:00:00Z",
            "offset": 2,
            "limit": 1,
            "max_unexpected_examples": 0,
        },
        context,
    )
    assert [item.security_id for item in paged.expected_pairs] == ["CCC"]
    assert paged.next_offset is None
    assert paged.unexpected_eligible_pairs == []
    assert paged.omitted_unexpected_examples == 2
    assert paged.missing_expected_pairs == 3
    none_expected = _run(
        coverage,
        {
            "observations": [
                _observation("s", "AAA", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
            ],
            "decision_time": "2026-01-01T10:00:00Z",
        },
        context,
    )
    assert none_expected.assessment == "no_expected_pairs"
    assert none_expected.complete_expected_coverage is None
    assert none_expected.unexpected_eligible_pair_count == 1
    all_missing = _run(
        coverage,
        {
            "expected_pairs": [{"source": "s", "security_id": "AAA"}],
            "observations": [],
            "decision_time": "2026-01-01T10:00:00Z",
        },
        context,
    )
    assert all_missing.expected_pairs[0].status == "missing"
    assert all_missing.eligible_record_count == 0
    tie = _run(
        coverage,
        {
            "expected_pairs": [{"source": "s", "security_id": "AAA"}],
            "observations": [
                _observation("s", "AAA", "2026-01-01T08:00:00Z", "2026-01-01T08:00:00Z"),
                _observation("s", "AAA", "2026-01-01T08:00:00Z", "2026-01-01T08:00:00Z"),
            ],
            "decision_time": "2026-01-01T10:00:00Z",
        },
        context,
    )
    assert tie.expected_pairs[0].selected_row_index == 0
    aware = datetime(2026, 1, 1, 10, tzinfo=UTC)
    via_datetime = _run(
        coverage,
        {
            "expected_pairs": [{"source": "s", "security_id": "AAA"}],
            "observations": [
                {
                    "source": "s",
                    "security_id": "AAA",
                    "event_time": datetime(2026, 1, 1, 9, tzinfo=UTC),
                    "available_time": datetime(2026, 1, 1, 9, tzinfo=UTC),
                }
            ],
            "decision_time": aware,
        },
        context,
    )
    assert via_datetime.expected_pairs[0].status == "covered"
    _rejects(
        coverage.Input,
        {
            "expected_pairs": [
                {"source": "s", "security_id": "AAA"},
                {"source": "s", "security_id": "AAA"},
            ],
            "decision_time": "2026-01-01T00:00:00Z",
        },
        "unique",
    )
    _rejects(coverage.Input, {"decision_time": LONG_CLOCK}, "64")
    _rejects(coverage.Input, {"decision_time": NAIVE_CLOCK}, "timezone")
    _rejects(coverage.Input, {"decision_time": 1_700_000_000}, "epoch")
    _rejects(coverage.Input, {"decision_time": OVERFLOW_CLOCK}, "UTC")
    _rejects(
        coverage.Input,
        {"decision_time": datetime(2026, 1, 1, tzinfo=_NullOffset())},
        "timezone",
    )
    _rejects(
        coverage.Input,
        {
            "observations": [_observation("s", "AAA", NAIVE_CLOCK, "2026-01-01T00:00:00Z")],
            "decision_time": "2026-01-01T00:00:00Z",
        },
        "timezone",
    )


# ---------------------------------------------------------------------------
# join_asof_observations
# ---------------------------------------------------------------------------


def _price(security: str, event: str, available: str, revision: str, value: int) -> dict[str, Any]:
    return {
        "security_id": security,
        "event_time": event,
        "available_time": available,
        "revision_id": revision,
        "source": "src",
        "values": {"px": value},
    }


def test_join_asof_latest_revision_lookback_and_page(context: OperationContext) -> None:
    observations = [
        _price("AAA", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "b", 1),
        _price("AAA", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "a", 1),
        _price("AAA", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "b", 1),
        _price("AAA", "2026-01-01T02:00:00Z", "2026-01-01T03:00:00Z", "a", 2),
        _price("BBB", "2026-01-01T01:00:00Z", "2026-01-01T01:00:00Z", "a", 9),
    ]
    result = _run(
        join_asof,
        {
            "observations": observations,
            "queries": [
                {"security_id": "AAA", "decision_time": "2026-01-01T01:00:00Z"},
                {"security_id": "BBB", "decision_time": "2026-01-01T00:30:00Z"},
                {"security_id": "AAA", "decision_time": "2026-01-01T04:00:00Z"},
                {"security_id": "CCC", "decision_time": "2026-01-01T04:00:00Z"},
            ],
            "max_event_age_seconds": 3600,
            "offset": 0,
            "limit": 3,
        },
        context,
    )
    assert result.matched_count == 1
    assert result.no_eligible_observation_count == 2
    assert result.outside_lookback_count == 1
    assert [item.query_row_index for item in result.joined] == [0, 1, 2]
    assert result.joined[0].observation_row_index == 0
    assert result.joined[0].event_age_seconds == pytest.approx(3600)
    assert result.joined[1].unmatched_reason == "no_eligible_observation"
    assert result.joined[2].unmatched_reason == "outside_lookback"
    assert result.joined[2].event_age_seconds == pytest.approx(2 * 3600)
    assert result.joined[2].observation_row_index is None
    assert result.next_offset == 3
    assert result.max_source_available_time_used == datetime(2026, 1, 1, tzinfo=UTC)
    assert [item.observation_row_index for item in result.selected_observations] == [0]
    visible = _run(
        join_asof,
        {
            "observations": observations,
            "queries": [
                {"security_id": "AAA", "decision_time": "2026-01-01T03:00:00Z"},
                {"security_id": "BBB", "decision_time": "2026-01-01T02:00:00Z"},
            ],
        },
        context,
    )
    assert visible.joined[0].observation_row_index == 3
    assert visible.joined[0].matched is True
    assert visible.joined[1].observation_row_index == 4
    assert visible.max_source_available_time_used == datetime(2026, 1, 1, 3, tzinfo=UTC)
    assert visible.next_offset is None
    aged = _run(
        join_asof,
        {
            "observations": [_price("AAA", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "a", 1)],
            "queries": [{"security_id": "AAA", "decision_time": "2026-01-01T00:00:00Z"}],
            "max_event_age_seconds": 0,
        },
        context,
    )
    assert aged.matched_count == 1
    assert aged.joined[0].event_age_seconds == 0
    later_availability_loses = _run(
        join_asof,
        {
            "observations": [
                _price("AAA", "2026-01-01T02:00:00Z", "2026-01-01T02:00:00Z", "a", 5),
                _price("AAA", "2026-01-01T01:00:00Z", "2026-01-01T04:00:00Z", "z", 1),
            ],
            "queries": [{"security_id": "AAA", "decision_time": "2026-01-01T04:00:00Z"}],
        },
        context,
    )
    assert later_availability_loses.joined[0].observation_row_index == 0
    older_availability = _run(
        join_asof,
        {
            "observations": [
                _price("AAA", "2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z", "a", 1),
                _price("AAA", "2026-01-01T03:00:00Z", "2026-01-01T01:00:00Z", "b", 2),
            ],
            "queries": [
                {"security_id": "AAA", "decision_time": "2026-01-01T02:00:00Z"},
                {"security_id": "AAA", "decision_time": "2026-01-01T04:00:00Z"},
            ],
        },
        context,
    )
    assert older_availability.joined[0].observation_row_index == 0
    assert older_availability.joined[1].observation_row_index == 1
    assert older_availability.max_source_available_time_used == datetime(2026, 1, 1, 2, tzinfo=UTC)


def test_join_asof_conflicts_duplicates_and_clocks(context: OperationContext) -> None:
    duplicate = _run(
        join_asof,
        {
            "observations": [
                _price("AAA", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "a", 1),
                _price("AAA", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00", "a", 1),
            ],
            "queries": [{"security_id": "AAA", "decision_time": "2026-01-01T01:00:00Z"}],
        },
        context,
    )
    assert duplicate.joined[0].observation_row_index == 0
    with pytest.raises(ValidationError, match="conflicting source or values"):
        join_asof.Input.model_validate(
            {
                "observations": [
                    _price("AAA", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "a", 1),
                    _price("AAA", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "b", 2),
                ],
                "queries": [{"security_id": "AAA", "decision_time": "2026-01-01T01:00:00Z"}],
            }
        )
    empty_obs = _run(
        join_asof,
        {
            "observations": [],
            "queries": [{"security_id": "AAA", "decision_time": "2026-01-01T00:00:00Z"}],
        },
        context,
    )
    assert empty_obs.no_eligible_observation_count == 1
    aware = _run(
        join_asof,
        {
            "observations": [
                {
                    "security_id": "AAA",
                    "event_time": datetime(2026, 1, 1, tzinfo=UTC),
                    "available_time": datetime(2026, 1, 1, tzinfo=UTC),
                    "revision_id": "a",
                    "source": "src",
                    "values": {"px": 1},
                }
            ],
            "queries": [
                {"security_id": "AAA", "decision_time": datetime(2026, 1, 1, 1, tzinfo=UTC)}
            ],
        },
        context,
    )
    assert aware.matched_count == 1
    _rejects(
        join_asof.Input,
        {
            "observations": [_price("AAA", NAIVE_CLOCK, "2026-01-01T00:00:00Z", "a", 1)],
            "queries": [{"security_id": "AAA", "decision_time": "2026-01-01T01:00:00Z"}],
        },
        "timezone",
    )
    _rejects(
        join_asof.Input,
        {
            "observations": [],
            "queries": [{"security_id": "AAA", "decision_time": 1_700_000_000}],
        },
        "timezone-aware",
    )
    _rejects(
        join_asof.Input,
        {
            "observations": [],
            "queries": [{"security_id": "AAA", "decision_time": OVERFLOW_CLOCK}],
        },
        "UTC",
    )
    _rejects(
        join_asof.Input,
        {
            "observations": [],
            "queries": [
                {"security_id": "AAA", "decision_time": datetime(2026, 1, 1, tzinfo=_NullOffset())}
            ],
        },
        "timezone",
    )


# ---------------------------------------------------------------------------
# audit_revision_conflicts
# ---------------------------------------------------------------------------


def _record(
    security: str = "AAA",
    event: str = "2026-01-01T00:00:00Z",
    revision: str = "r1",
    available: str = "2026-01-01T00:00:00Z",
    source: str = "src",
    values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "security_id": security,
        "event_time": event,
        "revision_id": revision,
        "available_time": available,
        "source": source,
        "values": {"px": 1} if values is None else values,
    }


def test_revision_conflicts_are_typed_and_bounded(context: OperationContext) -> None:
    records = [
        _record(),
        _record(values={"px": 1, "flag": True}),
        _record(source="other"),
        _record(available="2026-01-01T01:00:00Z"),
        _record(values={"flag": True, "px": 1}),
        _record(values={"px": 2}),
        _record(revision="solo"),
        _record(event="2026-01-01T05:00:00+05:00", revision="tz"),
        _record(event="2026-01-01T00:00:00+00:00", revision="tz"),
    ]
    result = _run(
        revisions,
        {"records": records, "max_diagnostics": 2, "max_indices_per_group": 2},
        context,
    )
    assert result.revision_group_count == 3
    assert result.conflicting_revision_groups == 1
    assert result.exact_duplicate_only_groups == 1
    assert result.exact_duplicate_rows == 2
    assert result.conflict_free is False
    assert result.unique_and_consistent is False
    assert result.conflict_field_counts == {
        "available_time": 1,
        "source": 1,
        "values": 1,
    }
    assert result.findings[0].status == "conflicting_revision"
    assert result.findings[0].omitted_row_indices == 4
    assert len(result.findings[0].row_indices) == 2
    assert result.omitted_findings == 0
    capped = _run(revisions, {"records": records, "max_diagnostics": 1}, context)
    assert capped.omitted_findings == 1
    assert capped.findings[0].status == "conflicting_revision"
    clean = _run(revisions, {"records": [_record(), _record(revision="other")]}, context)
    assert clean.conflict_free is True
    assert clean.unique_and_consistent is True
    assert clean.findings == []
    repeats = _run(
        revisions,
        {"records": [_record(), _record(), _record(), _record()], "max_indices_per_group": 2},
        context,
    )
    assert repeats.exact_duplicate_only_groups == 1
    assert repeats.unique_and_consistent is False
    assert repeats.conflict_free is True
    assert repeats.findings[0].omitted_row_indices == 2
    assert repeats.findings[0].unique_variant_count == 1
    values_only = _run(
        revisions,
        {"records": [_record(values={"px": 1}), _record(values={"px": True})]},
        context,
    )
    assert values_only.findings[0].conflict_fields == ["values"]
    clocks = _run(
        revisions,
        {
            "records": [
                _record(available="2026-01-01T05:00:00+05:00"),
                _record(available="2026-01-01T00:00:00Z"),
            ]
        },
        context,
    )
    assert clocks.conflict_free is True
    assert clocks.findings[0].status == "exact_duplicates_only"
    assert clocks.findings[0].conflict_fields == []
    suppressed = _run(
        revisions,
        {
            "records": [
                _record(revision="c1"),
                _record(revision="c1", source="other"),
                _record(revision="c2"),
                _record(revision="c2", values={"px": 9}),
            ],
            "max_diagnostics": 1,
        },
        context,
    )
    assert suppressed.conflicting_revision_groups == 2
    assert len(suppressed.findings) == 1
    assert suppressed.omitted_findings == 1


def test_revision_conflict_clock_failures() -> None:
    _rejects(revisions.Input, {"records": [_record(event=NAIVE_CLOCK)]}, "timezone")
    _rejects(revisions.Input, {"records": [_record(event=1_700_000_000)]}, "timezone-aware")
    _rejects(revisions.Input, {"records": [_record(available=OVERFLOW_CLOCK)]}, "UTC")
    _rejects(
        revisions.Input,
        {"records": [_record(event=datetime(2026, 1, 1, tzinfo=_NullOffset()))]},
        "timezone",
    )
    parsed = revisions.Input.model_validate(
        {
            "records": [
                {
                    **_record(),
                    "event_time": datetime(2026, 1, 1, tzinfo=UTC),
                    "available_time": datetime(2026, 1, 1, tzinfo=UTC),
                }
            ]
        }
    )
    assert parsed.records[0].event_time == datetime(2026, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# audit_monotonic_sequences
# ---------------------------------------------------------------------------


def _point(group: str, clock: str, value: float | int) -> dict[str, object]:
    return {"group": group, "event_time": clock, "value": value}


def test_monotonic_directions_duplicates_and_limits(context: OperationContext) -> None:
    rows = [
        _point("up", "2026-01-01T00:00:00Z", 1),
        _point("up", "2026-01-01T01:00:00Z", 1),
        _point("up", "2026-01-01T00:30:00Z", 0),
        _point("up", "2026-01-01T05:00:00+05:00", 2),
        _point("down", "2026-01-01T02:00:00Z", 3),
        _point("down", "2026-01-01T03:00:00Z", 2),
        _point("one", "2026-01-01T00:00:00Z", 5),
    ]
    result = _run(
        monotonic,
        {"observations": rows, "direction": "increasing", "strict": True, "max_diagnostics": 2},
        context,
    )
    groups = {item.group: item for item in result.groups}
    assert groups["up"].event_time_reversals == 2
    assert groups["up"].duplicate_event_rows == 1
    assert groups["up"].numeric_direction_violations == 2
    assert groups["up"].passed is False
    assert groups["down"].numeric_direction_violations == 1
    assert groups["one"].passed is None
    assert groups["one"].comparable_pairs == 0
    assert result.passed is False
    assert result.assessment == "failed"
    assert result.unassessed_groups == 1
    assert len(result.diagnostics) == 2
    assert result.omitted_diagnostics == result.diagnostic_count - 2
    allowed = _run(
        monotonic,
        {
            "observations": [
                _point("g", "2026-01-01T00:00:00Z", 1),
                _point("g", "2026-01-01T01:00:00Z", 2),
                _point("g", "2026-01-01T00:00:00+00:00", 3),
            ],
            "allow_duplicate_event_times": True,
        },
        context,
    )
    assert allowed.passed is False
    assert allowed.groups[0].duplicate_event_rows == 1
    duplicate_only = _run(
        monotonic,
        {
            "observations": [
                _point("g", "2026-01-01T00:00:00Z", 1),
                _point("g", "2026-01-01T01:00:00Z", 2),
                _point("g", "2026-01-01T02:00:00Z", 1),
            ],
            "allow_duplicate_event_times": True,
            "max_diagnostics": 0,
        },
        context,
    )
    # The third value decreases, so the sequence still fails; duplicates are not the cause.
    assert duplicate_only.violation_count == 1
    assert duplicate_only.diagnostics == []
    assert duplicate_only.omitted_diagnostics == duplicate_only.diagnostic_count
    nonstrict_equal = _run(
        monotonic,
        {
            "observations": [
                _point("g", "2026-01-01T00:00:00Z", 1),
                _point("g", "2026-01-01T01:00:00Z", 1),
            ]
        },
        context,
    )
    assert nonstrict_equal.passed is True
    assert nonstrict_equal.assessment == "passed"
    assert nonstrict_equal.numeric_direction_violations == 0
    strict_equal = _run(
        monotonic,
        {
            "observations": [
                _point("g", "2026-01-01T00:00:00Z", 1.0),
                _point("g", "2026-01-01T01:00:00Z", 1),
            ],
            "strict": True,
        },
        context,
    )
    assert strict_equal.numeric_direction_violations == 1
    decreasing = _run(
        monotonic,
        {
            "observations": [
                _point("g", "2026-01-01T00:00:00Z", 3),
                _point("g", "2026-01-01T01:00:00Z", 2),
                _point("g", "2026-01-01T02:00:00Z", 2),
            ],
            "direction": "decreasing",
        },
        context,
    )
    assert decreasing.passed is True
    decreasing_strict = _run(
        monotonic,
        {
            "observations": [
                _point("g", "2026-01-01T00:00:00Z", 3),
                _point("g", "2026-01-01T01:00:00Z", 4),
            ],
            "direction": "decreasing",
            "strict": True,
        },
        context,
    )
    assert decreasing_strict.numeric_direction_violations == 1
    empty = _run(monotonic, {"observations": []}, context)
    assert empty.assessment == "no_comparable_pairs"
    assert empty.passed is None
    assert empty.group_count == 0
    reversal_only = _run(
        monotonic,
        {
            "observations": [
                _point("g", "2026-01-01T02:00:00Z", 1),
                _point("g", "2026-01-01T01:00:00Z", 2),
            ]
        },
        context,
    )
    assert reversal_only.event_time_reversals == 1
    assert reversal_only.numeric_direction_violations == 0
    duplicate_allowed_pass = _run(
        monotonic,
        {
            "observations": [
                _point("g", "2026-01-01T00:00:00Z", 1),
                _point("g", "2026-01-01T01:00:00Z", 2),
                _point("g", "2026-01-01T00:00:00Z", 3),
            ],
            "allow_duplicate_event_times": True,
        },
        context,
    )
    assert duplicate_allowed_pass.groups[0].duplicate_event_rows == 1
    assert any(
        item.code == "duplicate_event_time" and item.is_violation is False
        for item in duplicate_allowed_pass.diagnostics
    )


def test_monotonic_clock_and_group_bounds() -> None:
    _rejects(monotonic.Input, {"observations": [_point("g", NAIVE_CLOCK, 1)]}, "timezone")
    _rejects(
        monotonic.Input,
        {"observations": [{"group": "g", "event_time": 1_700_000_000, "value": 1}]},
        "datetime",
    )
    _rejects(monotonic.Input, {"observations": [_point("g", "1700000000", 1)]}, "ISO calendar")
    _rejects(monotonic.Input, {"observations": [_point("g", LONG_CLOCK, 1)]}, "64")
    _rejects(monotonic.Input, {"observations": [_point("g", OVERFLOW_CLOCK, 1)]}, "UTC")
    _rejects(
        monotonic.Input,
        {
            "observations": [
                {
                    "group": "g",
                    "event_time": datetime(2026, 1, 1, tzinfo=_NullOffset()),
                    "value": 1,
                }
            ]
        },
        "timezone",
    )
    accepted = monotonic.Input.model_validate(
        {
            "observations": [
                {
                    "group": "g",
                    "event_time": datetime(2026, 1, 1, 5, tzinfo=timezone(timedelta(hours=5))),
                    "value": 1,
                }
            ]
        }
    )
    assert accepted.observations[0].event_time == datetime(2026, 1, 1, tzinfo=UTC)
    too_many = [_point(f"g{index}", "2026-01-01T00:00:00Z", 1) for index in range(257)]
    _rejects(monotonic.Input, {"observations": too_many}, "256 groups")


# ---------------------------------------------------------------------------
# summarize_ingestion_latency
# ---------------------------------------------------------------------------


def _lag(source: str, available: str, ingested: str) -> dict[str, str]:
    return {"source": source, "available_time": available, "ingested_time": ingested}


def test_ingestion_latency_quantiles_bins_and_pages(context: OperationContext) -> None:
    result = _run(
        latency,
        {
            "observations": [
                _lag("b", "2026-01-01T00:00:10Z", "2026-01-01T00:00:00Z"),
                _lag("a", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
                _lag("a", "2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z"),
                _lag("a", "2026-01-01T00:00:00Z", "2026-01-01T00:00:30Z"),
                _lag("a", "2026-01-01T00:00:00Z", "2026-01-01T00:02:00Z"),
            ],
            "quantiles": [0.0, 0.5, 1.0],
            "nonnegative_bin_edges_seconds": [0.0, 10.0, 60.0],
            "source_offset": 0,
            "source_limit": 1,
        },
        context,
    )
    assert result.overall.negative_lag_count == 1
    assert result.overall.zero_lag_count == 1
    assert result.overall.positive_lag_count == 3
    assert result.overall.signed.minimum_seconds == -10
    assert result.overall.signed.maximum_seconds == 120
    assert result.overall.signed.mean_seconds == pytest.approx(30.0)
    assert result.overall.signed.quantiles[1].seconds == pytest.approx(10.0)
    assert result.overall.nonnegative.count == 4
    assert result.overall.nonnegative.quantiles[0].seconds == 0
    assert result.overall.histogram[0].count == 1
    assert result.overall.histogram[0].upper_seconds_exclusive == 0
    # 0 lands in [0, 10); 10 and 30 land in [10, 60); 120 lands in the open last bin.
    assert [item.count for item in result.overall.histogram] == [1, 1, 2, 1]
    assert result.overall.histogram[-1].upper_seconds_exclusive is None
    assert result.source_count == 2
    assert result.sources[0].source == "a"
    assert result.next_source_offset == 1
    page = _run(
        latency,
        {
            "observations": [
                _lag("b", "2026-01-01T00:00:10Z", "2026-01-01T00:00:00Z"),
                _lag("a", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            ],
            "quantiles": [1.0],
            "nonnegative_bin_edges_seconds": [0.0],
            "source_offset": 5,
            "source_limit": 1,
        },
        context,
    )
    assert page.sources == []
    assert page.next_source_offset is None
    negatives = _run(
        latency,
        {
            "observations": [
                _lag("a", "2026-01-01T00:00:05Z", "2026-01-01T00:00:00Z"),
                _lag("a", "2026-01-01T00:00:09Z", "2026-01-01T00:00:00Z"),
            ],
            "quantiles": [0.0, 1.0],
            "nonnegative_bin_edges_seconds": [0.0, 1.0],
        },
        context,
    )
    assert negatives.overall.nonnegative.count == 0
    assert negatives.overall.nonnegative.mean_seconds is None
    assert negatives.overall.nonnegative.quantiles == []
    assert negatives.overall.histogram[1].count == 0
    assert negatives.overall.signed.population_std_seconds == pytest.approx(2.0)
    single = _run(
        latency,
        {
            "observations": [_lag("a", "2026-01-01T00:00:00Z", "2026-01-01T00:00:04Z")],
            "quantiles": [0.0, 1.0],
        },
        context,
    )
    assert single.overall.signed.population_std_seconds == 0
    assert single.overall.signed.quantiles[0].seconds == 4
    assert single.overall.signed.quantiles[1].seconds == 4
    aware = _run(
        latency,
        {
            "observations": [
                {
                    "source": "a",
                    "available_time": datetime(2026, 1, 1, tzinfo=UTC),
                    "ingested_time": datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC),
                }
            ]
        },
        context,
    )
    assert aware.overall.signed.mean_seconds == pytest.approx(3)


def test_ingestion_latency_clock_and_edge_failures() -> None:
    base = {"observations": [_lag("a", "2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z")]}
    _rejects(latency.Input, {**base, "quantiles": [0.5, 0.5]}, "strictly increasing")
    _rejects(latency.Input, {**base, "nonnegative_bin_edges_seconds": [1.0, 2.0]}, "start at zero")
    _rejects(
        latency.Input, {**base, "nonnegative_bin_edges_seconds": [0.0, 0.0]}, "strictly increasing"
    )
    _rejects(
        latency.Input,
        {"observations": [_lag("a", NAIVE_CLOCK, "2026-01-01T00:00:00Z")]},
        "timezone",
    )
    _rejects(
        latency.Input,
        {"observations": [_lag("a", 1_700_000_000, "2026-01-01T00:00:00Z")]},
        "timezone-aware",
    )
    _rejects(latency.Input, {"observations": [_lag("a", OVERFLOW_CLOCK, OVERFLOW_CLOCK)]}, "UTC")
    _rejects(
        latency.Input,
        {
            "observations": [
                {
                    "source": "a",
                    "available_time": datetime(2026, 1, 1, tzinfo=_NullOffset()),
                    "ingested_time": "2026-01-01T00:00:00Z",
                }
            ]
        },
        "timezone",
    )


# ---------------------------------------------------------------------------
# time_weighted_mean
# ---------------------------------------------------------------------------


def _hold(event: str, available: str, value: float) -> dict[str, object]:
    return {"event_time": event, "available_time": available, "value": value}


def test_time_weighted_mean_identity_and_late_revisions(context: OperationContext) -> None:
    result = _run(
        twmean,
        {
            "observations": [
                _hold("2025-12-31T23:59:00Z", "2025-12-31T23:59:00Z", 10),
                _hold("2025-12-31T23:00:00Z", "2025-12-31T23:59:30Z", 99),
                _hold("2026-01-01T00:00:30Z", "2026-01-01T00:00:30Z", 40),
                _hold("2026-01-01T00:02:00Z", "2026-01-01T00:02:00Z", 7),
            ],
            "query_time": "2026-01-01T00:01:00Z",
            "lookback_seconds": 60,
        },
        context,
    )
    assert result.time_weighted_mean == pytest.approx(25)
    assert result.integral_value_seconds == pytest.approx(1500)
    assert result.positive_duration_segments == 2
    assert result.initial_observation_index == 0
    assert result.contributing_observation_indices == [0, 2]
    assert result.rows_not_activated_before_query == 1
    assert result.coverage_fraction == 1
    assert result.covered_seconds == 60
    held = _run(
        twmean,
        {
            "observations": [
                _hold("2025-12-31T23:00:00Z", "2025-12-31T23:00:00Z", 1),
                _hold("2025-12-31T23:59:00Z", "2026-01-01T00:00:10Z", 5),
            ],
            "query_time": "2026-01-01T00:01:00Z",
            "lookback_seconds": 60,
        },
        context,
    )
    assert held.integral_value_seconds == pytest.approx(10 * 1 + 50 * 5)
    assert held.time_weighted_mean == pytest.approx(260 / 60)
    assert held.contributing_observation_indices == [0, 1]
    skipped_old = _run(
        twmean,
        {
            "observations": [
                _hold("2025-12-31T23:00:00Z", "2025-12-31T23:00:00Z", 4),
                _hold("2026-01-01T00:00:20Z", "2026-01-01T00:00:20Z", 7),
                _hold("2026-01-01T00:00:10Z", "2026-01-01T00:00:40Z", 100),
            ],
            "query_time": "2026-01-01T00:01:00Z",
            "lookback_seconds": 60,
        },
        context,
    )
    assert skipped_old.contributing_observation_indices == [0, 1]
    assert skipped_old.integral_value_seconds == pytest.approx(20 * 4 + 40 * 7)
    zero_duration = _run(
        twmean,
        {
            "observations": [
                _hold("2025-12-31T23:00:00Z", "2025-12-31T23:00:00Z", 1),
                _hold("2026-01-01T00:00:10Z", "2026-01-01T00:00:20Z", 2),
                _hold("2026-01-01T00:00:15Z", "2026-01-01T00:00:20Z", 3),
            ],
            "query_time": "2026-01-01T00:01:00Z",
            "lookback_seconds": 60,
        },
        context,
    )
    assert zero_duration.positive_duration_segments == 2
    assert zero_duration.contributing_observation_indices == [0, 2]
    assert zero_duration.integral_value_seconds == pytest.approx(20 * 1 + 40 * 3)
    at_query = _run(
        twmean,
        {
            "observations": [
                _hold("2025-12-31T23:00:00Z", "2025-12-31T23:00:00Z", 8),
                _hold("2026-01-01T00:01:00Z", "2026-01-01T00:01:00Z", 100),
            ],
            "query_time": "2026-01-01T00:01:00Z",
            "lookback_seconds": 60,
        },
        context,
    )
    assert at_query.time_weighted_mean == pytest.approx(8)
    assert at_query.rows_not_activated_before_query == 1
    with pytest.raises(ValueError, match="no observable initial"):
        _run(
            twmean,
            {
                "observations": [_hold("2026-01-01T00:00:30Z", "2026-01-01T00:00:30Z", 1)],
                "query_time": "2026-01-01T00:01:00Z",
                "lookback_seconds": 60,
            },
            context,
        )


def test_time_weighted_mean_clock_failures() -> None:
    payload = {
        "observations": [_hold("2025-12-31T23:00:00Z", "2025-12-31T23:00:00Z", 1)],
        "query_time": "2026-01-01T00:00:00Z",
        "lookback_seconds": 60,
    }
    _rejects(twmean.Input, {**payload, "query_time": NAIVE_CLOCK}, "timezone")
    _rejects(twmean.Input, {**payload, "query_time": 1_700_000_000}, "timezone-aware")
    _rejects(twmean.Input, {**payload, "query_time": OVERFLOW_CLOCK}, "UTC")
    _rejects(
        twmean.Input,
        {**payload, "query_time": datetime(2026, 1, 1, tzinfo=_NullOffset())},
        "timezone",
    )
    _rejects(
        twmean.Input,
        {
            "observations": [
                _hold("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", 1),
                _hold("2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", 2),
            ],
            "query_time": "2026-01-01T00:02:00Z",
            "lookback_seconds": 30,
        },
        "unique",
    )
    _rejects(
        twmean.Input,
        {
            "observations": [_hold("2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z", 1)],
            "query_time": "0001-01-02T00:00:00Z",
            "lookback_seconds": 31_536_000,
        },
        "supported datetime",
    )
    via_datetime = twmean.Input.model_validate(
        {
            "observations": [
                {
                    "event_time": datetime(2025, 12, 31, tzinfo=UTC),
                    "available_time": datetime(2025, 12, 31, tzinfo=UTC),
                    "value": 1.0,
                }
            ],
            "query_time": datetime(2026, 1, 1, tzinfo=UTC),
            "lookback_seconds": 60,
        }
    )
    assert via_datetime.observations[0].value == 1.0


# ---------------------------------------------------------------------------
# select_asof_revisions
# ---------------------------------------------------------------------------


def test_select_asof_tie_break_future_events_and_page(context: OperationContext) -> None:
    records = [
        _record(revision="r1", available="2026-01-01T00:00:00Z", values={"px": 1}),
        _record(revision="r2", available="2026-01-01T01:00:00Z", values={"px": 2}),
        _record(revision="r0", available="2026-01-01T01:00:00Z", values={"px": 2}),
        _record(revision="r2", available="2026-01-01T01:00:00Z", values={"px": 2}),
        _record(event="2026-01-02T00:00:00Z", available="2026-01-01T00:00:00Z", revision="future"),
        _record(available="2026-01-03T00:00:00Z", revision="late"),
        _record(security="BBB", revision="b", values={"px": 4}),
    ]
    result = _run(
        asof,
        {
            "records": records,
            "decision_time": "2026-01-01T02:00:00Z",
            "require_event_by_decision": False,
            "offset": 0,
            "limit": 2,
        },
        context,
    )
    assert result.unavailable_rows == 1
    assert result.future_event_rows_excluded == 0
    assert result.selected_event_count == 3
    assert result.superseded_or_duplicate_rows == result.eligible_rows - 3
    chosen = result.selected[0]
    assert chosen.record.security_id == "AAA"
    assert chosen.record.revision_id == "r2"
    assert chosen.input_row_index == 1
    assert chosen.eligible_vintage_rows == 4
    assert result.next_offset == 2
    assert result.selected[1].record.security_id == "AAA"
    completed = _run(
        asof,
        {
            "records": records,
            "decision_time": "2026-01-01T02:00:00Z",
            "require_event_by_decision": True,
            "offset": 3,
        },
        context,
    )
    assert completed.future_event_rows_excluded == 1
    assert completed.selected_event_count == 2
    assert completed.selected == []
    assert completed.next_offset is None
    same_clock = _run(
        asof,
        {
            "records": [
                _record(revision="a", values={"px": 1}),
                _record(revision="b", values={"px": 1}),
            ],
            "decision_time": "2026-01-01T01:00:00Z",
        },
        context,
    )
    assert same_clock.selected[0].record.revision_id == "b"
    assert same_clock.selected[0].eligible_vintage_rows == 2


def test_select_asof_conflicts_and_clocks() -> None:
    with pytest.raises(ValidationError, match="conflicting source or values"):
        asof.Input.model_validate(
            {
                "records": [
                    _record(values={"px": 1}),
                    _record(values={"px": 2}, revision="other"),
                ],
                "decision_time": "2026-01-01T01:00:00Z",
            }
        )
    _rejects(asof.Input, {"records": [_record()], "decision_time": NAIVE_CLOCK}, "timezone")
    _rejects(asof.Input, {"records": [_record()], "decision_time": 1_700_000_000}, "timezone-aware")
    _rejects(asof.Input, {"records": [_record()], "decision_time": OVERFLOW_CLOCK}, "UTC")
    _rejects(
        asof.Input,
        {"records": [_record()], "decision_time": datetime(2026, 1, 1, tzinfo=_NullOffset())},
        "timezone",
    )
    _rejects(
        asof.Input,
        {"records": [_record(event=NAIVE_CLOCK)], "decision_time": "2026-01-01T00:00:00Z"},
        "timezone",
    )
    parsed = asof.Input.model_validate(
        {
            "records": [_record()],
            "decision_time": datetime(2026, 1, 1, 1, tzinfo=UTC),
        }
    )
    assert parsed.decision_time == datetime(2026, 1, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# missingness and association
# ---------------------------------------------------------------------------


def test_missingness_runs_rates_and_scope(context: OperationContext) -> None:
    rows = [
        {},
        {"x": None, "y": 1},
        {"x": 1, "y": 1},
        {"x": "", "y": 1},
        {"x": " ", "y": None},
        {"x": 0, "y": 1},
        {"x": None, "y": 1},
    ]
    result = _run(
        missingness,
        {"rows": rows, "columns": ["x", "y"], "empty_string_is_missing": False},
        context,
    )
    by_name = {item.column: item for item in result.columns}
    assert by_name["x"].absent_count == 1
    assert by_name["x"].null_count == 2
    assert by_name["x"].empty_string_count == 1
    assert by_name["x"].effective_missing_count == 3
    assert by_name["x"].leading_missing_run == 2
    assert by_name["x"].longest_missing_run == 2
    assert by_name["x"].trailing_missing_run == 1
    assert by_name["x"].absent_rate == pytest.approx(1 / 7)
    assert result.no_missing_values is False
    assert result.incomplete_row_count == 4
    assert result.complete_row_count == 3
    assert result.assessment == "observed"
    assert result.effective_missing_cell_rate == pytest.approx(5 / 14)
    as_missing = _run(
        missingness,
        {"rows": rows, "empty_string_is_missing": True, "max_incomplete_examples": 1},
        context,
    )
    x_profile = next(item for item in as_missing.columns if item.column == "x")
    assert x_profile.effective_missing_count == 4
    assert as_missing.incomplete_row_indices == [0]
    assert as_missing.omitted_incomplete_examples == as_missing.incomplete_row_count - 1
    silent = _run(
        missingness,
        {"rows": [{"a": 1}, {"a": None}], "max_incomplete_examples": 0},
        context,
    )
    assert silent.incomplete_row_indices == []
    assert silent.omitted_incomplete_examples == 1
    assert silent.columns[0].trailing_missing_run == 1
    assert silent.columns[0].leading_missing_run == 0
    no_rows = _run(missingness, {"rows": [], "columns": ["a"]}, context)
    assert no_rows.assessment == "no_observations"
    assert no_rows.columns[0].absent_rate is None
    assert no_rows.complete_row_count == 0
    assert no_rows.no_missing_values is None
    no_columns = _run(missingness, {"rows": [{}, {}]}, context)
    assert no_columns.assessment == "no_columns"
    assert no_columns.complete_row_count is None
    assert no_columns.column_count == 0
    none_at_all = _run(missingness, {"rows": []}, context)
    assert none_at_all.assessment == "no_observations"
    assert none_at_all.complete_row_count is None
    complete = _run(missingness, {"rows": [{"a": True, "b": 1.5}, {"a": False, "b": 0.0}]}, context)
    assert complete.no_missing_values is True
    assert complete.complete_row_rate == 1
    scoped = _run(missingness, {"rows": [{"a": 1, "b": None}], "columns": ["a"]}, context)
    assert scoped.no_missing_values is True
    assert scoped.column_count == 1


def test_missingness_bounds() -> None:
    _rejects(missingness.Input, {"rows": [{f"f{i}": 1 for i in range(65)}]}, "64 fields")
    _rejects(missingness.Input, {"rows": _wide_rows(3907)}, "250000")
    blocks = [{f"n{block}_{index}": 1 for index in range(60)} for block in range(5)]
    _rejects(missingness.Input, {"rows": blocks}, "256 distinct")
    _rejects(missingness.Input, {"columns": ["a", "a"]}, "distinct")


def test_missingness_association_phi_and_empty(context: OperationContext) -> None:
    positive = _run(
        association,
        {
            "rows": [{}, {}, {"a": 1, "b": 1}, {"a": 1, "b": 1}],
            "columns": ["a", "b"],
        },
        context,
    )
    pair = positive.associations[0]
    assert pair.both_missing == 2
    assert pair.neither_missing == 2
    assert pair.jaccard == pytest.approx(1.0)
    assert pair.phi == pytest.approx(1)
    assert pair.left_missing_given_right_missing == pytest.approx(1)
    negative = _run(
        association,
        {
            "rows": [
                {"a": None, "b": 1},
                {"a": None, "b": 1},
                {"a": 1, "b": None},
                {"a": 1, "b": ""},
            ],
            "columns": ["a", "b"],
            "empty_string_is_missing": True,
        },
        context,
    )
    assert negative.associations[0].phi == pytest.approx(-1)
    assert negative.columns[0].null_count == 2
    assert negative.columns[1].empty_string_count == 1
    constant = _run(
        association,
        {"rows": [{}, {}], "columns": ["a", "b"]},
        context,
    )
    assert constant.associations[0].phi is None
    assert constant.associations[0].jaccard == pytest.approx(1)
    blanks = _run(
        association,
        {"rows": [], "columns": ["a", "b"], "empty_string_is_missing": False},
        context,
    )
    assert blanks.assessment == "no_observations"
    assert blanks.columns[0].missing_rate is None
    assert blanks.associations[0].phi is None
    assert blanks.associations[0].jaccard is None
    assert blanks.associations[0].joint_missing_rate is None
    assert blanks.associations[0].left_missing_given_right_missing is None
    kept = _run(
        association,
        {"rows": [{"a": "", "b": " "}], "columns": ["a", "b"], "empty_string_is_missing": False},
        context,
    )
    assert kept.columns[0].missing_count == 0
    assert kept.columns[0].empty_string_count == 1
    assert kept.associations[0].neither_missing == 1
    one_sided = _run(
        association,
        {"rows": [{"a": None, "b": 1}, {"a": 1, "b": 1}], "columns": ["a", "b"]},
        context,
    )
    assert one_sided.associations[0].right_missing_given_left_missing == pytest.approx(0.0)
    assert one_sided.associations[0].left_missing_given_right_missing is None


def test_missingness_association_bounds() -> None:
    _rejects(association.Input, {"columns": ["a", "a"]}, "distinct")
    _rejects(
        association.Input, {"rows": [{f"f{i}": 1 for i in range(65)}], "columns": ["a", "b"]}, "64"
    )
    _rejects(
        association.Input,
        {"rows": _wide_rows(3907), "columns": ["f00", "f01"]},
        "250000",
    )


# ---------------------------------------------------------------------------
# spectral_summary, permutation entropy, rolling features, bipower, rank
# ---------------------------------------------------------------------------


def test_spectral_summary_identities_and_statuses(context: OperationContext) -> None:
    nyquist = _run(
        spectral,
        {"values": [1.0, -1.0, 1.0, -1.0], "sample_interval_seconds": 1.0, "detrend": "none"},
        context,
    )
    assert nyquist.peak_frequency_hz == pytest.approx(0.5)
    assert nyquist.integrated_total_power == pytest.approx(1.0)
    assert nyquist.spectral_status == "finite"
    assert nyquist.entropy_status == "finite"
    assert nyquist.normalized_spectral_entropy is not None
    assert nyquist.positive_power_underflow is False
    interior = _run(
        spectral,
        {"values": [1.0, 0.0, -1.0, 0.0], "sample_interval_seconds": 1.0, "detrend": "none"},
        context,
    )
    assert interior.peak_frequency_hz == pytest.approx(0.25)
    constant = _run(
        spectral,
        {"values": [2.0, 2.0, 2.0, 2.0], "sample_interval_seconds": 1.0, "detrend": "mean"},
        context,
    )
    assert constant.spectral_status == "constant_input"
    assert constant.peak_frequency_hz is None
    assert constant.entropy_status == "no_positive_frequency_power"
    assert constant.integrated_positive_frequency_power == 0
    zeros = _run(
        spectral,
        {"values": [0.0, 0.0, 0.0], "sample_interval_seconds": 2.0, "detrend": "none"},
        context,
    )
    assert zeros.spectral_status == "constant_input"
    assert zeros.frequency_resolution_hz == pytest.approx(1 / 6)
    single_bin = _run(
        spectral,
        {"values": [0.0, 1.0], "sample_interval_seconds": 1.0, "detrend": "mean"},
        context,
    )
    assert single_bin.entropy_status == "single_positive_bin"
    assert single_bin.normalized_spectral_entropy is None
    assert single_bin.positive_frequency_bin_count == 1
    assert single_bin.peak_period_seconds == pytest.approx(2.0)
    underflow = _run(
        spectral,
        {
            "values": [1e-180, -1e-180, 1e-180, -1e-180],
            "sample_interval_seconds": 1.0,
            "detrend": "none",
        },
        context,
    )
    assert underflow.positive_power_underflow is True
    assert underflow.peak_frequency_hz == pytest.approx(0.5)
    paged = _run(
        spectral,
        {
            "values": [1.0, -1.0, 1.0, -1.0],
            "sample_interval_seconds": 1.0,
            "detrend": "mean",
            "offset": 0,
            "limit": 1,
        },
        context,
    )
    assert len(paged.bins) == 1
    assert paged.has_more is True
    assert paged.bins[0].index == 0
    beyond = _run(
        spectral,
        {
            "values": [1.0, -1.0, 1.0, -1.0],
            "sample_interval_seconds": 1.0,
            "offset": 10,
            "limit": 1,
        },
        context,
    )
    assert beyond.bins == []
    assert beyond.has_more is False
    ramp = _run(
        spectral,
        {"values": [1.0, 2.0, 3.0, 4.0], "sample_interval_seconds": 1.0, "detrend": "mean"},
        context,
    )
    assert ramp.integrated_positive_frequency_power > 0
    assert ramp.spectral_status == "finite"


def test_permutation_entropy_ties_delays_and_thresholds(context: OperationContext) -> None:
    rising = _run(
        entropy,
        {
            "values": [1.0, 2.0, 3.0, 4.0],
            "embedding_dimension": 3,
            "delay": 1,
            "minimum_patterns": 20,
        },
        context,
    )
    assert rising.used_embeddings == 2
    assert rising.entropy_nats == pytest.approx(0.0)
    assert rising.normalized_entropy == pytest.approx(0.0)
    assert rising.sample_status == "below_threshold"
    assert rising.patterns[0].ordinal_positions == [0, 1, 2]
    assert rising.patterns[0].probability == pytest.approx(1.0)
    short = _run(entropy, {"values": [1.0], "embedding_dimension": 3, "delay": 2}, context)
    assert short.sample_status == "no_embeddings"
    assert short.entropy_nats is None
    assert short.total_embeddings == 0
    dropped = _run(
        entropy,
        {
            "values": [1.0, 1.0, 2.0],
            "embedding_dimension": 2,
            "tie_policy": "drop",
            "minimum_patterns": 1,
        },
        context,
    )
    assert dropped.tied_embeddings == 1
    assert dropped.discarded_embeddings == 1
    assert dropped.used_embeddings == 1
    assert dropped.sample_status == "meets_threshold"
    assert dropped.patterns[0].ordinal_positions == [0, 1]
    stable = _run(
        entropy,
        {
            "values": [2.0, 2.0, 1.0],
            "embedding_dimension": 2,
            "tie_policy": "stable",
            "minimum_patterns": 1,
            "pattern_limit": 1,
        },
        context,
    )
    assert stable.discarded_embeddings == 0
    assert stable.used_embeddings == 2
    assert stable.observed_pattern_count == 2
    assert stable.omitted_pattern_count == 1
    with pytest.raises(ValueError, match="tied values"):
        _run(
            entropy,
            {"values": [1.0, 1.0], "embedding_dimension": 2, "tie_policy": "reject"},
            context,
        )
    none_used = _run(
        entropy,
        {"values": [1.0, 1.0, 1.0], "embedding_dimension": 2, "tie_policy": "drop"},
        context,
    )
    assert none_used.sample_status == "no_eligible_patterns"
    assert none_used.used_embeddings == 0
    delayed = _run(
        entropy,
        {
            "values": [1.0, 9.0, 2.0, 8.0, 0.0],
            "embedding_dimension": 2,
            "delay": 2,
            "minimum_patterns": 1,
        },
        context,
    )
    assert delayed.total_embeddings == 3
    assert delayed.used_embeddings == 3
    assert {tuple(item.ordinal_positions) for item in delayed.patterns} >= {(0, 1), (1, 0)}


def test_rolling_autocorrelation_identity_and_variance(context: OperationContext) -> None:
    perfect = _run(autocorr, {"values": [1.0, 2.0, 3.0], "window": 3, "lag": 1}, context)
    assert perfect.status == ["warmup", "warmup", "finite"]
    assert perfect.correlations[2] == pytest.approx(1.0)
    assert perfect.pairs_per_full_window == 2
    negative = _run(autocorr, {"values": [1.0, 3.0, 2.0], "window": 3, "lag": 1}, context)
    assert negative.correlations[2] == pytest.approx(-1.0)
    flat = _run(autocorr, {"values": [5.0, 5.0, 5.0], "window": 3, "lag": 1}, context)
    assert flat.status[-1] == "zero_variance"
    assert flat.correlations[-1] is None
    left_flat = _run(autocorr, {"values": [1.0, 1.0, 2.0], "window": 3, "lag": 1}, context)
    assert left_flat.status[-1] == "zero_variance"
    right_flat = _run(autocorr, {"values": [1.0, 2.0, 2.0], "window": 3, "lag": 1}, context)
    assert right_flat.status[-1] == "zero_variance"
    _rejects(
        autocorr.Input,
        {"values": [1.0, 2.0, 3.0], "window": 3, "lag": 2, "minimum_pairs": 2},
        "minimum_pairs",
    )
    shifted = _run(
        autocorr,
        {"values": [1e12, 1e12 + 1.0, 1e12 + 2.0], "window": 3, "lag": 1},
        context,
    )
    assert shifted.correlations[-1] == pytest.approx(1.0)


def test_rolling_mad_even_odd_zero_and_overflow(context: OperationContext) -> None:
    odd = _run(mad, {"values": [1.0, 5.0, 3.0], "window": 3}, context)
    assert odd.zscore_status[0] == "warmup"
    assert odd.medians[2] == pytest.approx(3.0)
    assert odd.median_absolute_deviations[2] == pytest.approx(2.0)
    assert odd.robust_zscores[2] == pytest.approx(0.0)
    even = _run(mad, {"values": [1.0, 3.0], "window": 2}, context)
    assert even.medians[1] == pytest.approx(2.0)
    assert even.median_absolute_deviations[1] == pytest.approx(1.0)
    assert even.robust_zscores[1] == pytest.approx(1.0)
    zero = _run(mad, {"values": [4.0, 4.0], "window": 2}, context)
    assert zero.zscore_status[-1] == "zero_mad"
    assert zero.robust_zscores[-1] is None
    lone = _run(mad, {"values": [4.0], "window": 1}, context)
    assert lone.zscore_status == ["zero_mad"]
    overflow = _run(mad, {"values": [0.0, 5e-324, 1e100], "window": 3}, context)
    assert overflow.zscore_status[-1] == "numeric_overflow"
    assert overflow.robust_zscores[-1] is None
    assert overflow.normal_consistency_factor_applied is False


def test_bipower_variation_identity(context: OperationContext) -> None:
    result = _run(bipower, {"returns": [1.0, -2.0, 0.0, 3.0]}, context)
    assert result.realized_variation == pytest.approx(14.0)
    assert result.adjacent_absolute_product_sum == pytest.approx(2.0)
    assert result.bipower_variation == pytest.approx(math.pi)
    assert result.applied_finite_sample_factor == pytest.approx(1.0)
    assert result.bipower_status == "finite"
    corrected = _run(
        bipower,
        {"returns": [1.0, -2.0, 0.0, 3.0], "finite_sample_policy": "n_over_n_minus_1"},
        context,
    )
    assert corrected.applied_finite_sample_factor == pytest.approx(4 / 3)
    assert corrected.bipower_variation == pytest.approx(math.pi * 4 / 3)
    zeros = _run(bipower, {"returns": [0.0, 0.0]}, context)
    assert zeros.realized_variation == 0
    assert zeros.bipower_variation == 0
    single = _run(bipower, {"returns": [5.0]}, context)
    assert single.bipower_status == "insufficient_pairs"
    assert single.bipower_variation is None
    assert single.realized_variation == pytest.approx(25.0)
    assert single.adjacent_pair_count == 0
    one_sided = _run(bipower, {"returns": [0.0, 2.0, 0.0]}, context)
    assert one_sided.adjacent_absolute_product_sum == 0


def test_rolling_rank_ties_and_warmup(context: OperationContext) -> None:
    result = _run(rank, {"values": [1.0, 3.0, 3.0, 2.0], "window": 3}, context)
    assert result.percentiles[0] is None
    assert result.percentiles[1] is None
    assert result.percentiles[2] == pytest.approx((1 + (2 + 1) / 2) / 3)
    assert result.percentiles[3] == pytest.approx(1.0 / 3.0)
    unique = _run(rank, {"values": [1.0, 2.0, 3.0], "window": 3}, context)
    assert unique.percentiles[-1] == pytest.approx(1.0)
    assert _run(rank, {"values": [5.0], "window": 1}, context).percentiles == [1.0]
    assert result.tie_convention.startswith("average_one_based")


# ---------------------------------------------------------------------------
# resolve_security_identity and bar integrity
# ---------------------------------------------------------------------------


def _mapping(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "mapping_id": "m1",
        "security_id": "SEC",
        "ticker": "AAA",
        "exchange": "XNAS",
        "valid_from": "2020-01-01T00:00:00Z",
        "valid_to": None,
        "available_time": "2020-01-01T00:00:00Z",
        "revision_id": "r1",
        "source": "src",
    }
    row.update(updates)
    return row


def _query(
    ticker: str = "AAA", exchange: str = "XNAS", decision: str = "2021-01-01T00:00:00Z"
) -> dict[str, str]:
    return {"ticker": ticker, "exchange": exchange, "decision_time": decision}


def test_resolve_identity_revisions_ambiguity_and_budget(context: OperationContext) -> None:
    rows = [
        _mapping(security_id="OLD", available_time="2020-01-01T00:00:00Z", revision_id="r1"),
        _mapping(security_id="NEW", available_time="2020-06-01T00:00:00Z", revision_id="r2"),
        _mapping(mapping_id="m2", security_id="OTHER", ticker="BBB"),
        _mapping(
            mapping_id="m3",
            security_id="TEMP",
            ticker="CCC",
            valid_to="2020-12-31T00:00:00Z",
        ),
        _mapping(security_id="HIDDEN", available_time="2022-01-01T00:00:00Z", revision_id="r9"),
    ]
    result = _run(
        identity,
        {
            "mappings": rows,
            "queries": [
                _query(decision="2020-03-01T00:00:00Z"),
                _query(decision="2020-09-01T00:00:00Z"),
                _query(ticker="BBB"),
                _query(ticker="CCC", decision="2020-12-31T00:00:00Z"),
                _query(ticker="ZZZ"),
            ],
            "offset": 1,
            "limit": 2,
        },
        context,
    )
    assert result.resolved_count == 3
    assert result.missing_count == 2
    assert [item.status for item in result.resolutions] == ["resolved", "resolved"]
    assert result.resolutions[0].security_id == "NEW"
    assert result.next_offset == 3
    assert result.work_units > 0
    conflict = _run(
        identity,
        {
            "mappings": [
                _mapping(source="left", available_time="2020-01-01T00:00:00Z"),
                _mapping(source="right", available_time="2020-01-01T00:00:00Z"),
            ],
            "queries": [_query()],
        },
        context,
    )
    assert conflict.resolutions[0].status == "ambiguous"
    assert conflict.resolutions[0].conflicting_mapping_ids == 1
    assert conflict.ambiguous_count == 1
    two_names = _run(
        identity,
        {
            "mappings": [
                _mapping(mapping_id="m1", security_id="A"),
                _mapping(mapping_id="m2", security_id="B"),
            ],
            "queries": [_query()],
            "max_candidates_per_query": 2,
        },
        context,
    )
    assert two_names.resolutions[0].status == "ambiguous"
    assert two_names.resolutions[0].candidate_security_count == 2
    tie = _run(
        identity,
        {
            "mappings": [
                _mapping(mapping_id="late", revision_id="a", available_time="2020-02-01T00:00:00Z"),
                _mapping(
                    mapping_id="early", revision_id="z", available_time="2020-02-01T00:00:00Z"
                ),
            ],
            "queries": [_query()],
        },
        context,
    )
    assert tie.resolutions[0].status == "resolved"
    assert tie.resolutions[0].selected_mapping_row_index == 1
    before_vintage = _run(
        identity,
        {
            "mappings": [_mapping(available_time="2021-06-01T00:00:00Z")],
            "queries": [_query(decision="2021-01-01T00:00:00Z")],
        },
        context,
    )
    assert before_vintage.resolutions[0].status == "missing"
    crowded = _run(
        identity,
        {
            "mappings": [
                _mapping(mapping_id=f"m{index}", security_id="SAME") for index in range(4)
            ],
            "queries": [_query()],
            "max_candidates_per_query": 2,
        },
        context,
    )
    assert crowded.resolutions[0].status == "resolved"
    assert crowded.resolutions[0].omitted_candidate_rows == 2
    with pytest.raises(ValueError, match="split the query batch"):
        _run(
            identity,
            {
                "mappings": [
                    _mapping(mapping_id="m1"),
                    _mapping(mapping_id="m2", security_id="OTHER"),
                ],
                "queries": [_query()],
                "work_budget": 1,
            },
            context,
        )
    with pytest.raises(ValueError, match="candidate mapping work exceeds"):
        _run(
            identity,
            {
                "mappings": [_mapping(mapping_id="a"), _mapping(mapping_id="b")],
                "queries": [_query()],
                "work_budget": 2,
            },
            context,
        )
    with pytest.raises(ValueError, match="revision inspection"):
        _run(
            identity,
            {
                "mappings": [
                    _mapping(revision_id="a", available_time="2020-01-01T00:00:00Z"),
                    _mapping(revision_id="b", available_time="2020-01-01T00:00:00Z"),
                ],
                "queries": [_query()],
                "work_budget": 2,
            },
            context,
        )


def test_resolve_identity_clock_failures() -> None:
    _rejects(
        identity.Input, {"mappings": [], "queries": [_query(decision=NAIVE_CLOCK)]}, "timezone"
    )
    _rejects(
        identity.Input, {"mappings": [], "queries": [_query(decision="1700000000")]}, "isoformat"
    )
    _rejects(
        identity.Input,
        {"mappings": [_mapping()], "queries": [_query(decision=OVERFLOW_CLOCK)]},
        "UTC",
    )
    _rejects(
        identity.Input,
        {
            "mappings": [],
            "queries": [_query(decision=datetime(2026, 1, 1, tzinfo=_NullOffset()).isoformat())],
        },
        "timezone|clock",
    )
    _rejects(
        identity.Input,
        {"mappings": [_mapping(valid_to="2020-01-01T00:00:00Z")], "queries": [_query()]},
        "strictly after",
    )
    _rejects(
        identity.Input,
        {"mappings": [_mapping(valid_from=1_700_000_000)], "queries": [_query()]},
        "timezone-aware",
    )
    parsed = identity.Input.model_validate(
        {
            "mappings": [
                {
                    **_mapping(),
                    "valid_from": datetime(2020, 1, 1, tzinfo=UTC),
                    "available_time": datetime(2020, 1, 1, tzinfo=UTC),
                }
            ],
            "queries": [
                {
                    "ticker": "AAA",
                    "exchange": "XNAS",
                    "decision_time": datetime(2021, 1, 1, tzinfo=UTC),
                }
            ],
        }
    )
    assert parsed.queries[0].decision_time == datetime(2021, 1, 1, tzinfo=UTC)
    past = _run(
        identity,
        {
            "mappings": [_mapping()],
            "queries": [_query(), _query(ticker="BBB"), _query(ticker="CCC")],
            "offset": 5,
        },
        OperationContext(workspace_root=Path(".")),
    )
    assert past.resolutions == []
    assert past.resolved_count == 1
    assert past.missing_count == 2


def test_bar_integrity_nonfinite_float_and_envelope(context: OperationContext) -> None:
    rogue = bars.Bar.model_construct(open=float("nan"), high=2.0, low=1.0, close=1.5, volume=1.0)
    request = bars.Input.model_construct(bars=[rogue], max_diagnostics=10)
    result = bars.execute(request, context)
    assert result.violation_counts["nonfinite_value"] == 1
    assert result.passed is False
    audited = _run(
        bars,
        {
            "bars": [
                {"open": 10, "high": 12, "low": 9, "close": 11, "volume": 0},
                {"open": 0, "high": 2, "low": 3, "close": 1, "volume": -1},
                {"open": "Infinity", "high": "nope", "low": None, "close": 1, "volume": "2"},
            ],
            "max_diagnostics": 2,
        },
        context,
    )
    assert audited.valid_rows == 1
    assert audited.invalid_rows == 2
    assert audited.violation_counts["nonpositive_price"] == 1
    assert audited.violation_counts["negative_volume"] == 1
    assert audited.violation_counts["inverted_range"] == 1
    assert audited.violation_counts["nonfinite_value"] == 1
    assert audited.violation_counts["nonnumeric_value"] == 2
    assert audited.violation_counts["missing_value"] == 1
    assert len(audited.diagnostics) == 2
    assert audited.omitted_diagnostics == audited.violation_count - 2
    edges = _run(
        bars,
        {"bars": [{"open": 9, "high": 12, "low": 9, "close": 12, "volume": 0}]},
        context,
    )
    assert edges.passed is True
    outside = _run(
        bars,
        {"bars": [{"open": 8, "high": 3, "low": 1, "close": 2, "volume": 1}]},
        context,
    )
    assert outside.violation_counts == {"outside_envelope": 1}
    assert outside.passed is False
