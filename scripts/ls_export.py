#!/usr/bin/env python3
"""Merge Carrot Cake Lab (Label Studio) review decisions into review/decisions.json.

The review Space `Hebrish/hebrew-latin-span-review` kept decisions in browser localStorage and you downloaded
`decisions.json`. Label Studio projects "Hebrish: span review", "Hebrish: boundary review" and "Hebrish: record audit"
replace it; this script writes the same records the page's `record()` wrote, keyed by `data.key`, and merges them into
the existing file (existing keys are only overwritten by a newer Label Studio decision). Then run
`scripts/build_splits.py` as before.

API on the LAN: http://10.0.0.2:8140 (the public host is behind Cloudflare Access). Token: LABEL_STUDIO_ADMIN_TOKEN.

    with-secrets uv run scripts/ls_export.py --project <span_pid> --project <boundary_pid> --project <record_pid>
        # dry run: prints counts, writes /tmp/decisions.ls.json
    ... --apply   # rewrite review/decisions.json in place

Record shapes (from the Space's index.html):
    span      {kind:'span', include, hebrew, at}
    boundary  {kind:'boundary', term, parts, short_term, choice, include, hebrew, at}
    record    {kind:'record', terms, at}
    can't decide (any kind) {kind, term, include:null, deferred:true, at}
Extra, Label Studio only: span `corrected_span: [start, end]` when the Hebrew region was moved off the candidate
(build_splits.py ignores it; it flags candidates whose boundary was wrong). Skipped tasks are not written.
`at` = the annotation's updated_at as JS toISOString() (UTC, milliseconds, Z).
"""
# /// script
# dependencies = ["requests"]
# ///
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "review" / "decisions.json"
LS_URL = os.environ.get("CCL_API_URL", "http://10.0.0.2:8140")


def js_iso(s: str) -> str:
    d = datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"


def choice(ann: dict, name: str):
    for r in ann["result"]:
        if r.get("from_name") == name and r.get("type") == "choices":
            return (r["value"].get("choices") or [None])[0]
    return None


def record(t: dict, ann: dict, warn) -> dict | None:
    """Label Studio task data + annotation -> the Space's decision record."""
    at = js_iso(ann["updated_at"])
    kind = t["kind"]
    if kind == "span":
        c = choice(ann, "decision")
        if c == "cant_decide":
            return {"kind": kind, "term": t.get("term"), "include": None, "deferred": True, "at": at}
        if c not in ("include", "exclude"):
            warn(f"{t['key']}: no decision"); return None
        rec = {"kind": "span", "include": c == "include", "hebrew": c == "include", "at": at}
        regions = [(r["value"]["start"], r["value"]["end"]) for r in ann["result"] if r.get("from_name") == "span"]
        if regions and regions != [(t["start"], t["end"])]:
            rec["corrected_span"] = list(regions[0]) if len(regions) == 1 else [list(x) for x in regions]
        return rec
    if kind == "boundary":
        c = choice(ann, "boundary")
        if c == "cant_decide":
            return {"kind": kind, "term": t.get("term"), "include": None, "deferred": True, "at": at}
        if c not in ("full", "short", "none"):
            warn(f"{t['key']}: no boundary choice"); return None
        return {"kind": "boundary", "term": t["term"], "parts": t.get("parts"), "short_term": t.get("short_term"),
                "choice": c, "include": c == "full", "hebrew": c == "full", "at": at}
    if kind == "record":
        if choice(ann, "status") == "cant_decide":
            return {"kind": kind, "term": t.get("term"), "include": None, "deferred": True, "at": at}
        raw = ""
        for r in ann["result"]:
            if r.get("from_name") == "terms":
                raw = ", ".join(r["value"].get("text") or [])
        terms = [s.strip() for s in raw.split(",") if s.strip()]
        if terms and choice(ann, "status") == "none_missed":
            warn(f"{t['key']}: 'None missed' ticked but terms typed; terms kept")
        return {"kind": "record", "terms": terms, "at": at}
    warn(f"{t['key']}: unknown kind {kind}"); return None


def export(pid: int) -> list[dict]:
    r = requests.get(f"{LS_URL}/api/projects/{pid}/export", params={"exportType": "JSON"},
                     headers={"Authorization": f"Token {os.environ['LABEL_STUDIO_ADMIN_TOKEN']}"}, timeout=120)
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, action="append", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    warn = lambda m: print("WARN", m, file=sys.stderr)
    decisions = json.loads(DECISIONS.read_text())
    new = 0
    for pid in a.project:
        for task in export(pid):
            anns = [x for x in task.get("annotations", []) if not x.get("was_cancelled")]
            if not anns:
                continue
            rec = record(task["data"], max(anns, key=lambda x: x["updated_at"]), warn)
            key = task["data"]["key"]
            if rec and (key not in decisions or decisions[key].get("at", "") < rec["at"]):
                decisions[key] = rec; new += 1
    text = json.dumps(decisions, indent=1, ensure_ascii=False)   # = JSON.stringify(decisions, null, 1)
    out = DECISIONS if a.apply else Path("/tmp/decisions.ls.json")
    out.write_text(text)
    print(json.dumps({"new_or_updated": new, "total": len(decisions), "written": str(out)}))


if __name__ == "__main__":
    main()
