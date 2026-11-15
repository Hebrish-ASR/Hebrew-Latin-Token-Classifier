"""Write-back for label scheme v2: Label Studio project 39 + LLM pre-annotations -> data/v2/labels.jsonl.

For each corpus sentence:
    reviewed       a human annotation exists in project 39 (latest one wins); "can't decide" is kept but flagged
    auto_accepted  no task in the project (not selected by ls_build_v2.py --select): the LLM spans are used as-is
    pending        a task exists but has no annotation yet: LLM spans, flagged pending

Project 47 verdicts (term-level, Daniel): "Plain English" drops that lemma's SHARED spans from every sentence
without a human annotation; "Hebrew" retags them HEB. Human-annotated sentences keep what the human drew.

Output rows: {id, split, text, status, spans: [{start, end, surface, tag, lang, hebrew_variant}], source}
name_type: null | brand | person | place; place_type (human only, when place): city | neighbourhood | road | kibbutz | moshav | region | landmark; compound, garbled: bool; term (+ term_matched) and lemma via lib/terms.py + data/term_links.csv;
variety (per sentence): null | anglo_israeli | yeshivish | jewish_american. lang defaults to "hebrew"; hebrew_variant defaults to "standard" when lang is hebrew, else null.
Label Studio choice values are capitalised (Yiddish, Ashkenazic); they are stored lower-case here.

    with-secrets uv run scripts/ls_export_v2.py                 # dry run: counts only
    with-secrets uv run scripts/ls_export_v2.py --apply         # write data/v2/labels.jsonl

Needs LABEL_STUDIO_ADMIN_TOKEN; API on the LAN (http://10.0.0.2:8140), since the public host is behind Cloudflare Access.
"""
from __future__ import annotations

import argparse, collections, json, os, pathlib, sys, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib.corpus import read_jsonl, write_jsonl  # noqa: E402
from lib.terms import link, lemma  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = os.environ.get("CCL_API_URL", "http://10.0.0.2:8140")


def export(pid: int) -> list[dict]:
    req = urllib.request.Request(f"{API}/api/projects/{pid}/export?exportType=JSON&download_all_tasks=true",
                                 headers={"Authorization": f"Token {os.environ['LABEL_STUDIO_ADMIN_TOKEN']}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


TAG = {"Hebrew": "HEB", "Shared": "SHARED"}   # display names; the config aliases them back to HEB / SHARED


def spans_from_result(result: list[dict]) -> tuple[list[dict], bool]:
    regions, extra, cant, variety, text_error = {}, collections.defaultdict(dict), False, None, False
    for item in result:
        v = item.get("value", {})
        if item["from_name"] == "tag":
            regions[item["id"]] = {"start": v["start"], "end": v["end"], "surface": v.get("text", ""),
                                   "tag": TAG.get(v["labels"][0], v["labels"][0])}
        elif item["from_name"] in ("lang", "hebrew_variant", "name_type", "place_type", "compound", "garbled") and v.get("choices"):
            extra[item["id"]][item["from_name"]] = v["choices"][0].lower()
        elif item["from_name"] == "sentence" and v.get("choices"):
            cant = cant or "can't decide" in v["choices"]
            text_error = "Text error" in v["choices"]
        elif item["from_name"] == "variety" and v.get("choices"):
            variety = v["choices"][0].lower().replace("-", "_")
    spans = []
    for rid, s in regions.items():
        lang = extra[rid].get("lang", "hebrew")
        s["lang"] = lang
        s["hebrew_variant"] = extra[rid].get("hebrew_variant", "standard") if lang == "hebrew" else None
        s["name_type"] = extra[rid].get("name_type")            # None = ordinary vocabulary
        s["place_type"] = extra[rid].get("place_type") if s["name_type"] == "place" else None
        s["compound"] = extra[rid].get("compound") == "compound"
        s["garbled"] = extra[rid].get("garbled") == "garbled spelling"
        spans.append(s)
    return sorted(spans, key=lambda s: s["start"]), cant, variety, text_error


def shared_verdicts(pid: int) -> dict[str, str]:
    """Project 47 "Hebrish: Shared or plain English?": term -> Shared | Plain English | Hebrew (latest annotation)."""
    out = {}
    for t in export(pid):
        anns = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
        if not anns:
            continue
        ann = max(anns, key=lambda x: x.get("updated_at") or x.get("created_at") or "")
        for item in ann["result"]:
            if item["from_name"] == "verdict" and item["value"].get("choices"):
                out[t["data"]["term"]] = item["value"]["choices"][0]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=int, default=39)
    p.add_argument("--pre", default="data/v2/preannotations.google-gemini-3-8-flash.v22.jsonl")
    p.add_argument("--shared-project", type=int, default=47)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    pre = {r["id"]: r for r in read_jsonl(ROOT / a.pre)}
    tasks = {t["data"]["id"]: t for t in export(a.project)}
    verdict = shared_verdicts(a.shared_project)
    out, stats = [], collections.Counter()
    for sid, r in pre.items():
        t = tasks.get(sid)
        anns = [x for x in (t or {}).get("annotations", []) if not x.get("was_cancelled")]
        if anns:
            ann = max(anns, key=lambda x: x.get("updated_at") or x.get("created_at") or "")
            spans, cant, variety, text_error = spans_from_result(ann["result"])
            status = "cant_decide" if cant else "reviewed"
            source = f"human:project{a.project}"
        else:
            spans = [{**s, "hebrew_variant": s.get("hebrew_variant") if s["lang"] == "hebrew" else None}
                     for s in r["spans"]]
            status = "pending" if t else "auto_accepted"
            source = f"llm:{r['model']}"
            variety = r.get("variety")
            text_error = False
        for sp in spans:                        # recomputed every export, so new data/term_links.csv rows apply
            sp["term"], sp["term_matched"] = link(sp["surface"])
            sp["lemma"] = lemma(sp["term"])
        # Shared-or-plain-English verdicts (project 47) override the tag of LLM spans; a human's own span wins.
        if not anns:
            kept = []
            for sp in spans:
                v = verdict.get(sp["lemma"]) if sp["tag"] == "SHARED" else None
                if v == "Plain English":
                    stats["dropped_plain_english"] += 1
                    continue
                if v == "Hebrew":
                    sp["tag"] = "HEB"; stats["retagged_heb"] += 1
                kept.append(sp)
            spans = kept
        stats[status] += 1
        out.append({"id": sid, "split": r["split"], "text": r["text"], "status": status, "variety": variety, "text_error": text_error, "spans": spans,
                    "source": source})
    print(json.dumps({"sentences": len(out), **stats}))
    if a.apply:
        write_jsonl(ROOT / "data" / "v2" / "labels.jsonl", out)
        print("wrote data/v2/labels.jsonl")


if __name__ == "__main__":
    main()
