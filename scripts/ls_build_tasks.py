#!/usr/bin/env python3
"""Build Carrot Cake Lab (Label Studio) tasks from the round-1 review queue.

Replaces the browser-only review Space `Hebrish/hebrew-latin-span-review` for the tasks still open in
`review/queue.json` (no decision in `review/decisions.json`). One Label Studio project per task kind:

    span      -> project "Hebrish: span review"      (candidate span pre-highlighted as a prediction)
    boundary  -> project "Hebrish: boundary review"
    record    -> project "Hebrish: record audit"

`term` tasks are not built: round 2 (`review/queue-terms.json`, 614 terms) is fully decided and round 1 has one open
term task. Order = queue order, as the Space served it. `data.key` is the decisions.json key (the join on the way back).

Usage:
    uv run scripts/ls_build_tasks.py --kind span --limit 15 -o /tmp/span.json
    then: ccl.py import <pid> /tmp/span.json [--confirm]   (Carrot-Cake-Lab-Plugin)
Write-back: scripts/ls_export.py.
"""
import argparse, html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def span_task(t: dict) -> dict:
    header = f"{t['risk']}: {t['note']}"
    return {"data": {**t, "header": header},
            "predictions": [{"model_version": "candidate", "result": [{
                "id": "cand", "from_name": "span", "to_name": "text", "type": "labels",
                "value": {"start": t["start"], "end": t["end"], "text": t["text"][t["start"]:t["end"]],
                          "labels": ["Hebrew"]}}]}]}


def boundary_task(t: dict) -> dict:
    surfaces = t.get("part_surfaces") or [t["short_surface"]]
    parts = " + ".join(f"<b>{html.escape(s)}</b>" for s in surfaces)
    what = "Both parts" if len(surfaces) > 1 else "Shorter"
    ex = "".join(f"<li>{html.escape(x)}</li>" for x in t.get("examples") or [])
    q = (f"<p style='font-size:1.2em'><b>{html.escape(t['surface'])}</b> as one span, or as {parts}?</p>"
         f"<p style='opacity:.7'>{html.escape(t['note'])}</p><ul>{ex}</ul>"
         f"<p style='font-size:.85em;opacity:.65'><b>Shorter / the parts</b> here means <b>{what}</b>. "
         "<b>Neither</b> = tag nothing; use it when the shorter piece is a fragment meaning something else "
         "on its own, like 'af ehad' (nobody) cut to 'ehad' (one).</p>")
    return {"data": {**t, "question_html": q}}


def record_task(t: dict) -> dict:
    return {"data": dict(t)}


BUILD = {"span": span_task, "boundary": boundary_task, "record": record_task}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=sorted(BUILD))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    queue = json.loads((ROOT / "review/queue.json").read_text())
    done = json.loads((ROOT / "review/decisions.json").read_text())
    todo = [t for t in queue if t["kind"] == a.kind and t["key"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    Path(a.out).write_text(json.dumps([BUILD[a.kind](t) for t in todo], ensure_ascii=False))
    print(json.dumps({"kind": a.kind, "tasks": len(todo), "out": a.out}))


if __name__ == "__main__":
    main()
