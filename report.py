"""Generate a self-contained, offline HTML audit report. No plotting dependency."""
import json
from pathlib import Path


def write_report(frame, metrics, out, provenance=None):
    out = Path(out)
    if provenance is None:
        source = out / "extraction.json"
        provenance = json.loads(source.read_text()) if source.exists() else {}
    # pandas emits strict JSON nulls for unknown labels and absent OOF predictions.
    rows = json.loads(frame.to_json(orient="records", double_precision=10))
    payload = {"rows": rows, "metrics": metrics, "provenance": provenance,
               "columns": list(frame.columns)}
    data = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    path = out / "report.html"
    path.write_text(template.replace("__DREAM_DATA__", data), encoding="utf-8")
    return path
