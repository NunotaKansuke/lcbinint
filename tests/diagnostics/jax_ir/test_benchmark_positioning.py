"""Small policy tests for the matched-accuracy positioning benchmark."""

from benchmark_positioning import select_first_passing


def test_resolution_selection_requires_the_complete_higher_resolution_tail():
    rows = [
        {"source_bins": 16, "passes": False},
        {"source_bins": 32, "passes": True},
        {"source_bins": 64, "passes": False},
        {"source_bins": 96, "passes": True},
        {"source_bins": 128, "passes": True},
    ]
    assert select_first_passing(rows)["source_bins"] == 96


def test_resolution_selection_rejects_an_unstable_final_tail():
    rows = [
        {"source_bins": 16, "passes": True},
        {"source_bins": 32, "passes": False},
    ]
    assert select_first_passing(rows) is None
