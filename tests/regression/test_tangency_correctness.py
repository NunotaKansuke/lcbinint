"""Regression corpus for native binary-lens tangency correctness."""

from __future__ import annotations

import subprocess
import sys


def test_native_tangency_arbiter_and_seed_invariance(tmp_path):
    output = tmp_path / "tangency-correctness.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.diagnostics.recal2026.tangency_correctness",
            "--check",
            "--output",
            str(output),
        ],
        check=True,
    )
    assert output.is_file()
