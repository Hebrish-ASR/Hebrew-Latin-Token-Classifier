"""Write-back for label scheme v2: Label Studio project 39 + LLM pre-annotations -> data/v2/labels.jsonl.

For each corpus sentence:
    reviewed       a human annotation exists in project 39 (latest one wins); "can't decide" is kept but flagged
    auto_accepted  no task in the project (not selected by ls_build_v2.py --select): the LLM spans are used as-is
    pending        a task exists but has no annotation yet: LLM spans, flagged pending

Output rows: {id, split, text, status, spans: [{start, end, surface, tag, lang, hebrew_variant}], source}
name_type: null | brand | person | place; compound: bool. lang defaults to "hebrew"; hebrew_variant defaults to "standard" when lang is hebrew, else null.
Label Studio choice values are capitalised (Yiddish, Ashkenazic); they are stored lower-case here.

    with-secrets uv run scripts/ls_export_v2.py                 # dry run: counts only
    with-secrets uv run scripts/ls_export_v2.py --apply         # write data/v2/labels.jsonl

Needs LABEL_STUDIO_ADMIN_TOKEN; API on the LAN (http://10.0.0.2:8140), since the public host is behind Cloudflare Access.
"""
from __future__ import annotations

import argparse, collections, json, os, pathlib, sys, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib.corpus import read_jsonl, write_jsonl  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = os.environ.get("CCL_API_URL", "http://10.0.0.2:8140")


def export(pid: int) -> list[dict]:
    req = urllib.request.Request(f"{API}/api/projects/{pid}/export?exportType=JSON&download_all_tasks=true",
                                 headers={"Authorization": f"Token {os.environ['LABEL_STUDIO_ADMIN_TOKEN']}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def spans_from_result(result: list[dict]) -> tuple[list[dict], bool]:
    regions, extra, cant = {}, collections.defaultdict(dict), False
    for item in result:
        v = item.get("value", {})
        if item["from_name"] == "tag":
            regions[item["id"]] = {"start": v["start"], "end": v["end"], "surface": v.get("text", ""),
                                   "tag": v["labels"][0]}
        elif item["from_name"] in ("lang", "hebrew_variant", "name_type", "compound") and v.get("choices"):
            extra[item["id"]][item["from_name"]] = v["choices"][0].lower()
        elif item["from_name"] == "sentence" and v.get("choices"):
            cant = True
    spans = []
    for rid, s in regions.items():
        lang = extra[rid].get("lang", "hebrew")
        s["lang"] = lang
        s["hebrew_variant"] = extra[rid].get("hebrew_variant", "standard") if lang == "hebrew" else None
        s["name_type"] = extra[rid].get("name_type")            # None = ordinary vocabulary
        s["compound"] = extra[rid].get("compound") == "compound"
        spans.append(s)
    return sorted(spans, key=lambda s: s["start"]), cant


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=int, default=39)
    p.add_argument("--pre", default="data/v2/preannotations.google-gemini-3-8-flash.v21.jsonl")
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    pre = {r["id"]: r for r in read_jsonl(ROOT / a.pre)}
    tasks = {t["data"]["id"]: t for t in export(a.project)}
    out, stats = [], collections.Counter()
    for sid, r in pre.items():
        t = tasks.get(sid)
        anns = [x for x in (t or {}).get("annotations", []) if not x.get("was_cancelled")]
        if anns:
            ann = max(anns, key=lambda x: x.get("updated_at") or x.get("created_at") or "")
            spans, cant = spans_from_result(ann["result"])
            status = "cant_decide" if cant else "reviewed"
            source = f"human:project{a.project}"
        else:
            spans = [{**s, "hebrew_variant": s.get("hebrew_variant") if s["lang"] == "hebrew" else None}
                     for s in r["spans"]]
            status = "pending" if t else "auto_accepted"
            source = f"llm:{r['model']}"
        stats[status] += 1
        out.append({"id": sid, "split": r["split"], "text": r["text"], "status": status, "spans": spans,
                    "source": source})
    print(json.dumps({"sentences": len(out), **stats}))
    if a.apply:
        write_jsonl(ROOT / "data" / "v2" / "labels.jsonl", out)
        print("wrote data/v2/labels.jsonl")


if __name__ == "__main__":
    main()
