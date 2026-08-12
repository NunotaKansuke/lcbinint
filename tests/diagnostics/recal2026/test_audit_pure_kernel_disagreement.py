"""Tests for the read-only pure-kernel disagreement audit."""

from __future__ import annotations

import copy

import pytest

from .audit_pure_kernel_disagreement import _markdown, audit_payload


def _inputs():
    lcbinint_value = 10.05
    target_vbm_value = 10.2
    stored_difference = abs(lcbinint_value - target_vbm_value) / target_vbm_value
    result = {
        "status": "completed",
        "case_id": 7,
        "profile": "linear",
        "target": 1.0e-2,
        "x": 0.125,
        "y": -0.25,
        "actual_d_over_rho": 0.6,
        "chosen_grid": ["cartesian"],
        "chosen_nbin": [10],
        "chosen_vbm_errors": [stored_difference],
        # Legacy benchmark name: an independent target-RelTol VBM value.
        "reference": [target_vbm_value],
        "vbm": {"timing_values": [10.19]},
        "grid": {
            "cartesian": {
                "samples": {
                    "0": {
                        "nbin": [5, 10, 20],
                        "magnification": [9.9, lcbinint_value, 10.01],
                    }
                }
            },
            "polar": {
                "samples": {
                    "0": {
                        "nbin": [5, 10, 20],
                        "magnification": [9.8, 10.04, 10.02],
                    }
                }
            },
        },
    }
    corpus_row = {
        "case_id": 7,
        "profile": "linear",
        "x": 0.125,
        "y": -0.25,
        "references": {
            "0": {"value": 10.0, "uncertainty": 2.0e-6, "status": "ok"}
        },
    }
    payload = {"reference_indices": [0], "results": [result]}
    return payload, [corpus_row]


def test_audit_reconstructs_selected_value_without_running_a_kernel():
    payload, corpus = _inputs()
    audit = audit_payload(payload, corpus)
    record = audit["epochs"][0]

    assert audit["mode"] == "posthoc_read_only_no_kernel_remeasurement"
    assert audit["self_check"]["reconstructed_stored_pairs"] == 1
    assert audit["self_check"]["max_absolute_residual"] == 0.0
    assert record["lcbinint_value"] == 10.05
    assert record["cartesian_tail_witness"] == 10.01
    assert record["polar_tail_witness"] == 10.02
    assert record["cross_engine_outside_nominal_band"] is True
    assert (
        record["tight_vbm_witness_band"]
        == "lcbinint_only_within_nominal_band"
    )


def test_audit_rejects_an_inconsistent_saved_pairwise_difference():
    payload, corpus = _inputs()
    inconsistent = copy.deepcopy(payload)
    inconsistent["results"][0]["chosen_vbm_errors"][0] += 1.0e-3

    with pytest.raises(ValueError, match="selected-value reconstruction"):
        audit_payload(inconsistent, corpus)


def test_summary_markdown_rows_match_the_header_width():
    payload, corpus = _inputs()
    markdown = _markdown(
        audit_payload(payload, corpus),
        {"reference_relative_levels": [1.0e-6, 1.0e-7]},
    )
    lines = markdown.splitlines()
    header_index = lines.index(next(
        line for line in lines if line.startswith("| profile | target | epochs")
    ))
    header = lines[header_index]
    row = lines[header_index + 2]

    assert header.count("|") == row.count("|")
    assert row.count("neither_within_nominal_band") == 0
