"""Build Label Studio tasks for label scheme v2: one task per sentence, LLM spans pre-drawn as predictions.

The reviewer adds missing spans, deletes wrong ones, retags HEB <-> SHARED, and sets lang / hebrew_variant on a
span only when it is an exception (default hebrew / standard: an unset choice means the default).

    uv run scripts/ls_build_v2.py data/v2/pilot.google-gemini-3-8-flash.jsonl -o /tmp/v2.json
    then: ccl.py create-project scripts/ls_v2_config.xml --title "Hebrish: sentence labels v2" --confirm
          ccl.py import <pid> /tmp/v2.json --confirm          (Carrot-Cake-Lab-Plugin)

Each task's data carries the v1 spans and a `differs` flag (LLM vs v1). Sort or filter on it in the Data Manager
to review disagreements first. Write-back: scripts/ls_export_v2.py (not written yet).
"""
from __future__ import annotations

import argparse, html, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib.corpus import read_jsonl  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def v1_spans() -> dict[str, list[dict]]:
    out = {}
    for split in ("train", "validation", "test"):
        for r in read_jsonl(ROOT / "data" / "corpus" / f"{split}.jsonl"):
            out[r["id"]] = r["spans"]
    return out


HELP = ("<div style='font-size:.8em;opacity:.65;margin:.3em 0 .6em'><b>HEB</b> romanized Hebrew not known in English "
        "(makolet) &middot; <b>SHARED</b> also an English word (Shabbat, kosher) &middot; place names: leave untagged. "
        "Add: press h or s, then select the word. Remove: click the span, Backspace. Click a span to mark an exception "
        "(Yiddish, Ashkenazic ...); blank means Hebrew / standard.</div>")


def task(row: dict, v1: list[dict]) -> dict:
    llm_set = {(s["start"], s["end"]) for s in row["spans"]}
    v1_set = {(s["start"], s["end"]) for s in v1}
    differs = llm_set != v1_set
    v1_txt = ", ".join(s["surface"] for s in v1) or "none"
    note = (f"<span style='color:#c92a2a'><b>Check:</b> differs from the old labels (old: {html.escape(v1_txt)})</span>"
            if differs else f"<span style='opacity:.6'>Matches the old labels ({html.escape(v1_txt)})</span>")
    note += HELP
    result = []
    for k, s in enumerate(row["spans"]):
        rid = f"r{k}"
        pos = {"start": s["start"], "end": s["end"], "text": s["surface"]}
        result.append({"id": rid, "from_name": "tag", "to_name": "text", "type": "labels",
                       "value": {**pos, "labels": [s["tag"]]}})
        if s["lang"] != "hebrew":
            result.append({"id": rid, "from_name": "lang", "to_name": "text", "type": "choices",
                           "value": {**pos, "choices": [s["lang"]]}})
        if s.get("hebrew_variant") not in (None, "standard"):
            result.append({"id": rid, "from_name": "hebrew_variant", "to_name": "text", "type": "choices",
                           "value": {**pos, "choices": [s["hebrew_variant"]]}})
    return {"data": {"id": row["id"], "split": row["split"], "text": row["text"], "note": note,
                     "differs": differs, "v1": v1_txt, "llm_model": row["model"]},
            "predictions": [{"model_version": row["model"], "result": result}]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("preannotations")
    p.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    v1 = v1_spans()
    rows = read_jsonl(pathlib.Path(a.preannotations))
    tasks = [task(r, v1.get(r["id"], [])) for r in rows]
    tasks.sort(key=lambda t: not t["data"]["differs"])       # disagreements first in import order
    pathlib.Path(a.out).write_text(json.dumps(tasks, ensure_ascii=False, indent=1))
    print(json.dumps({"tasks": len(tasks), "differs": sum(t["data"]["differs"] for t in tasks), "out": a.out}))


if __name__ == "__main__":
    main()
