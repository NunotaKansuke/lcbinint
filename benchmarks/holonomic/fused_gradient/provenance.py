"""Record the precise overlay, compiler and backend used by the experiment."""
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

destination = Path(sys.argv[1])
paths = list(Path("benchmarks/holonomic/fused_gradient").glob("*")) + [
    Path("build-seeded-events/seed_indexed_v30.o"),
    Path("build-standard-current/liblcbinint_magnification.a"),
    Path("build-standard-current/liblcbinint_lightcurve.a"),
    Path("evidence/holonomic/seeded_paper_qrho_20261014/input_snapshot.tsv"),
    Path("build-seeded-events/fused-gradient/baseline"),
    Path("build-seeded-events/fused-gradient/fused"),
]
def output(*args):
    return subprocess.check_output(args, text=True).strip()
data = {
    "checkout": str(Path.cwd()), "branch": output("git", "branch", "--show-current"),
    "base_commit": output("git", "rev-parse", "HEAD"),
    "host": platform.node(), "machine": platform.machine(),
    "compiler": output("g++", "--version").splitlines()[0],
    "repetitions": 3, "value_rtol": 1e-4, "value_atol": 1e-8,
    "timing": "fresh magnifier per repetition; caustic prebuild excluded; seed/index preparation included; seed set shared; independent workspace; alternating value/fused order",
    "gradient_precision_checked": False,
    "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()},
    "overlay": json.loads(Path("build-seeded-events/fused-gradient/include/lcbinint/magnification/holonomic/manifest.json").read_text()),
}
(destination / "provenance.json").write_text(json.dumps(data, indent=2) + "\n")
