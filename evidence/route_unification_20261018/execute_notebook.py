"""Execute this plain-Python notebook and capture its plot without local sockets."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
import nbformat

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "example/compare-vbm/lcbinint_vbm_light_curve_comparison.ipynb"
notebook = nbformat.read(path, as_version=4)
namespace = {"__name__": "__main__"}
count = 0
for cell in notebook.cells:
    if cell.cell_type != "code" or not cell.source.strip():
        continue
    count += 1
    stream, images = io.StringIO(), []
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        if "plt" in namespace:
            def show():
                for number in namespace["plt"].get_fignums():
                    image = io.BytesIO()
                    namespace["plt"].figure(number).savefig(image, format="png")
                    images.append(image.getvalue())
            namespace["plt"].show = show
        exec(compile(cell.source, str(path)+f":cell{count}", "exec"), namespace)
    cell.execution_count, cell.outputs = count, []
    if stream.getvalue():
        cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=stream.getvalue()))
    for raw in images:
        cell.outputs.append(nbformat.v4.new_output("display_data",
            data={"image/png": base64.b64encode(raw).decode()}, metadata={}))
        Path(__file__).with_name("notebook.png").write_bytes(raw)
    print("cell", count, "OK", stream.getvalue())
nbformat.write(notebook, path)
summary = dict(value_ms=namespace["lc_seconds"]*1e3,
               vbm_ms=namespace["vbm_seconds"]*1e3,
               gradient_ms=namespace["gradient_seconds"]*1e3,
               max_relative_vbm=float(namespace["np"].max(namespace["relative_error"])))
Path(__file__).with_name("notebook-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print("All code cells executed; figure captured. No socket-based Jupyter kernel used.")
