"""Build Label Studio tasks for label scheme v2: one task per sentence, LLM spans pre-drawn as predictions.

The reviewer adds missing spans, deletes wrong ones, retags HEB <-> SHARED, and sets lang / hebrew_variant on a
span only when it is an exception (default hebrew / standard: an unset choice means the default).

    uv run scripts/ls_build_v2.py data/v2/pilot.google-gemini-3-8-flash.jsonl -o /tmp/v2.json
    uv run scripts/ls_build_v2.py data/v2/preannotations.google-gemini-3-8-flash.jsonl --select --skip ids.txt -o /tmp/v2.json
        (--select: only what needs a human -- differs from v1, test split, 5% audit; the rest is auto-accepted)
    then: ccl.py create-project scripts/ls_v2_config.xml --title "Hebrish: sentence labels v2" --confirm
          ccl.py import <pid> /tmp/v2.json --confirm          (Carrot-Cake-Lab-Plugin)

Each task's data carries the v1 spans and a `differs` flag (LLM vs v1). Sort or filter on it in the Data Manager
to review disagreements first. Choice values are capitalised in Label Studio (Yiddish, Ashkenazic ...);
the corpus stores them lower-case, so the export lower-cases them. Write-back: scripts/ls_export_v2.py (not written yet).
"""
from __future__ import annotations

import argparse, html, json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib.corpus import read_jsonl  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def v1_spans() -> dict[str, list[dict]]:
    out = {}
    for split in ("train", "validation", "test"):
        for r in read_jsonl(ROOT / "data" / "corpus" / f"{split}.jsonl"):
            out[r["id"]] = r["spans"]
    return out


HELP = ("<div style='font-size:.8em;opacity:.65;margin:.3em 0 .6em'><b>Hebrew</b> (h): romanized Hebrew not known in "
        "English (makolet) &middot; <b>Shared</b> (s): also an English word (Shabbat, kosher). Add: press h or s, then select "
        "the word. Remove: click the span, Backspace. Fixed phrases (cheder madregot) are one span; a Hebrew word with an "
        "English ending is tagged whole (kibbutzniks). Click a span for names, form and exceptions; blank means the default. "
        "All correct: Ctrl+Enter.</div>")


VARIETY = {"anglo_israeli": "Anglo-Israeli", "yeshivish": "Yeshivish", "jewish_american": "Jewish-American"}


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
                           "value": {**pos, "choices": [s["lang"].capitalize()]}})
        if s.get("name_type"):
            result.append({"id": rid, "from_name": "name_type", "to_name": "text", "type": "choices",
                           "value": {**pos, "choices": [s["name_type"].capitalize()]}})
        if s.get("compound"):
            result.append({"id": rid, "from_name": "compound", "to_name": "text", "type": "choices",
                           "value": {**pos, "choices": ["Compound"]}})
        if s.get("garbled"):
            result.append({"id": rid, "from_name": "garbled", "to_name": "text", "type": "choices",
                           "value": {**pos, "choices": ["Garbled spelling"]}})
        if s.get("hebrew_variant") not in (None, "standard"):
            result.append({"id": rid, "from_name": "hebrew_variant", "to_name": "text", "type": "choices",
                           "value": {**pos, "choices": [s["hebrew_variant"].capitalize()]}})
    if row.get("variety"):
        result.append({"id": "variety", "from_name": "variety", "to_name": "text", "type": "choices",
                       "value": {"choices": [VARIETY[row["variety"]]]}})
    return {"data": {"id": row["id"], "split": row["split"], "text": row["text"], "note": note,
                     "differs": differs, "v1": v1_txt, "llm_model": row["model"]},
            "predictions": [{"model_version": row["model"], "result": result}]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("preannotations")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--select", action="store_true",
                   help="review queue only: LLM differs from v1, the whole test split, and an audit sample of the rest")
    p.add_argument("--audit", type=float, default=0.05, help="share of the remaining sentences to audit (default 5%%)")
    p.add_argument("--skip", help="file of sentence ids already in the project (one per line)")
    a = p.parse_args()
    v1 = v1_spans()
    rows = read_jsonl(pathlib.Path(a.preannotations))
    skip = set(pathlib.Path(a.skip).read_text().split()) if a.skip else set()
    tasks = [task(r, v1.get(r["id"], [])) for r in rows if r["id"] not in skip]
    why = {}
    if a.select:
        rng = random.Random(15112026)
        keep = []
        for t in tasks:
            d = t["data"]
            reason = "differs" if d["differs"] else "test" if d["split"] == "test" else (
                "audit" if rng.random() < a.audit else None)
            if reason:
                d["queue"] = reason; keep.append(t)
            why[reason or "auto_accepted"] = why.get(reason or "auto_accepted", 0) + 1
        tasks = keep
    order = {"differs": 0, "test": 1, "audit": 2}
    tasks.sort(key=lambda t: (order.get(t["data"].get("queue"), 0 if t["data"]["differs"] else 1)))
    pathlib.Path(a.out).write_text(json.dumps(tasks, ensure_ascii=False, indent=1))
    print(json.dumps({"tasks": len(tasks), "differs": sum(t["data"]["differs"] for t in tasks), "skipped": len(skip),
                      "by_queue": why, "out": a.out}))


if __name__ == "__main__":
    main()
