"""Annotate ahead of the reviewer: replicate Daniel's term and span decisions onto open Label Studio tasks.

A spiral: every decision Daniel makes in project 39 (or 47) is generalised, and any open sentence that his decisions
already settle is annotated by the bot user `term-verdicts` (LS user id 3, created 2026-11-16), so it leaves his queue.
Run it once, or with --watch to keep doing this while he labels.

Decisions learnt from human annotations (bot annotations never count as evidence):
    term verdict   per lemma (lib/terms.py): SHARED | HEB | DROPPED (an LLM span he deleted). Project 47 verdicts
                   (Shared / Hebrew / Plain English = DROPPED) win over project 39 votes. A lemma he has labelled two
                   different ways has no verdict (listed under "conflicts") and is left to him.
    span edit      an LLM span he redrew (tofes 17 -> tofes): keyed on the normalised LLM surface, stored as the
                   offset + length of his span inside it. Conflicting redraws of the same surface: no rule.

An open project 39 task (no annotation at all, queue "differs") is annotated when, after applying the span edits,
every LLM span's lemma has a verdict and every v1-only span's lemma is DROPPED. Spans with DROPPED are removed; the
others keep the LLM's attributes (lang, name_type ...) with the verdict's tag. Test-split and audit tasks are never
annotated ahead: they exist to check the LLM. Spans the LLM missed cannot be inferred, the same risk as the 3,117
auto-accepted sentences. Open project 47 terms with a project 39 verdict get the matching choice.

A bot annotation that no longer follows from the decisions (he changed his mind, a conflict appeared) is deleted, or
replaced. ls_export_v2.py writes bot-annotated sentences as status "extrapolated", and a human annotation always wins.

    with-secrets uv run scripts/ls_propagate.py                  # dry run: what it would annotate
    with-secrets uv run scripts/ls_propagate.py --apply
    with-secrets uv run scripts/ls_propagate.py --apply --watch 60   # repeat every 60 s while labelling

Needs LABEL_STUDIO_ADMIN_TOKEN; API on the LAN (http://10.0.0.2:8140).
"""
from __future__ import annotations

import argparse, collections, json, os, pathlib, sys, time, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib.corpus import read_jsonl  # noqa: E402
from lib.orthography import normalise  # noqa: E402
from lib.terms import link, lemma  # noqa: E402
from ls_build_v2 import task as build_task, v1_spans  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = os.environ.get("CCL_API_URL", "http://10.0.0.2:8140")
BOT = "term-verdicts"
TAG = {"Hebrew": "HEB", "Shared": "SHARED", "HEB": "HEB", "SHARED": "SHARED"}
V47 = {"Shared": "SHARED", "Hebrew": "HEB", "Plain English": "DROPPED"}
V47_BACK = {v: k for k, v in V47.items()}


def call(path: str, method: str = "GET", body=None):
    req = urllib.request.Request(f"{API}{path}", method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Token {os.environ['LABEL_STUDIO_ADMIN_TOKEN']}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def export(pid: int) -> list[dict]:
    return call(f"/api/projects/{pid}/export?exportType=JSON&download_all_tasks=true")


def lem(surface: str) -> str:
    return lemma(link(surface)[0])


def split_anns(t: dict, bot: int) -> tuple[list[dict], list[dict]]:
    anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
    return [a for a in anns if a["completed_by"] != bot], [a for a in anns if a["completed_by"] == bot]


def latest(anns: list[dict]) -> dict:
    return max(anns, key=lambda x: x.get("updated_at") or x.get("created_at") or "")


def learn(t39: list[dict], t47: list[dict], pre: dict, bot: int):
    votes, edits = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for t in t39:
        human, _ = split_anns(t, bot)
        if not human:
            continue
        text = t["data"]["text"]
        spans = [(r["value"]["start"], r["value"]["end"], TAG.get(r["value"]["labels"][0]))
                 for r in latest(human)["result"] if r["from_name"] == "tag"]
        for s, e, tag in spans:
            votes[lem(text[s:e])][tag] += 1
        for sp in pre[t["data"]["id"]]["spans"]:
            over = [(s, e) for s, e, _ in spans if s < sp["end"] and e > sp["start"]]
            if not over:
                votes[lem(sp["surface"])]["DROPPED"] += 1
            elif len(over) == 1 and over[0] != (sp["start"], sp["end"]):
                s, e = over[0]
                if s >= sp["start"] and e <= sp["end"]:          # shrunk inside the LLM span: a replicable edit
                    edits[normalise(sp["surface"])][(s - sp["start"], e - s)] += 1
    verdict = {k: next(iter(v)) for k, v in votes.items() if len(v) == 1}
    conflicts = {k: dict(v) for k, v in votes.items() if len(v) > 1}
    for t in t47:
        human, _ = split_anns(t, bot)
        if human:
            for r in latest(human)["result"]:
                if r["from_name"] == "verdict" and r["value"].get("choices", [""])[0] in V47:
                    verdict[t["data"]["term"]] = V47[r["value"]["choices"][0]]
                    conflicts.pop(t["data"]["term"], None)
    rules = {k: next(iter(v)) for k, v in edits.items() if len(v) == 1}
    return verdict, conflicts, rules


def resolve(row: dict, v1: list[dict], verdict: dict, rules: dict) -> list[dict] | None:
    """The spans Daniel's decisions give this sentence, or None if something is still undecided."""
    out = []
    for sp in row["spans"]:
        sp = dict(sp)
        rule = rules.get(normalise(sp["surface"]))
        if rule:
            off, n = rule
            sp["start"] += off; sp["end"] = sp["start"] + n
            sp["surface"] = row["text"][sp["start"]:sp["end"]]
        v = verdict.get(lem(sp["surface"]))
        if v is None:
            return None
        if v != "DROPPED":
            sp["tag"] = v
            out.append(sp)
    for s in v1:
        if not any(o["start"] < s["end"] and o["end"] > s["start"] for o in out) and verdict.get(lem(s["surface"])) != "DROPPED":
            if not any(o["start"] < s["end"] and o["end"] > s["start"] for o in row["spans"]):
                return None                                       # v1 span the LLM missed and he hasn't dropped
    return out


def canon(result: list[dict]) -> str:
    return json.dumps(sorted(json.dumps(r, sort_keys=True) for r in result))


def run(a, pre: dict, v1: dict, bot: int) -> dict:
    t39, t47 = export(a.project), export(a.shared_project)
    verdict, conflicts, rules = learn(t39, t47, pre, bot)
    stats, changes = collections.Counter(), []
    for t in t39:
        human, mine = split_anns(t, bot)
        if human:
            continue
        d = t["data"]
        row = pre[d["id"]]
        spans = resolve(row, v1.get(d["id"], []), verdict, rules) if d.get("queue") == "differs" else None
        want = None
        if spans is not None and (row["spans"] or v1.get(d["id"])):
            want = build_task({**row, "spans": spans}, v1.get(d["id"], []))["predictions"][0]["result"]
        have = latest(mine) if mine else None
        if have and (want is None or canon(have["result"]) != canon(want)):
            changes.append(("delete", have["id"], None)); stats["bot_withdrawn"] += 1
            have = None
        if want is not None and have is None:
            changes.append(("annotate", t["id"], want)); stats["annotate_39"] += 1
            if len([c for c in changes if c[0] == "annotate"]) <= 5:
                print("  39:", d["text"][:80], "->", [(s["surface"], s["tag"]) for s in spans])
        elif want is not None:
            stats["already_bot_39"] += 1
        else:
            stats["left_for_daniel_39"] += 1
    for t in t47:
        human, mine = split_anns(t, bot)
        if human:
            continue
        v = verdict.get(t["data"]["term"])
        want = [{"id": "v", "from_name": "verdict", "to_name": "term_html", "type": "choices",
                 "value": {"choices": [V47_BACK[v]]}}] if v else None
        have = latest(mine) if mine else None
        if have and (want is None or have["result"][0]["value"] != want[0]["value"]):
            changes.append(("delete", have["id"], None)); stats["bot_withdrawn"] += 1; have = None
        if want and have is None:
            changes.append(("annotate", t["id"], want)); stats["annotate_47"] += 1
    if a.apply:
        for op, ident, result in changes:
            if op == "delete":
                call(f"/api/annotations/{ident}/", "DELETE")
            else:
                call(f"/api/tasks/{ident}/annotations/", "POST", {"result": result, "completed_by": bot, "lead_time": 0})
    return {"at": time.strftime("%H:%M:%S"), "verdicts": len(verdict), "span_rules": len(rules),
            "conflicts": conflicts, "applied": a.apply, **stats}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=int, default=39)
    p.add_argument("--shared-project", type=int, default=47)
    p.add_argument("--pre", default="data/v2/preannotations.google-gemini-3-8-flash.v22.jsonl")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--watch", type=int, metavar="SECONDS", help="repeat every SECONDS until interrupted")
    a = p.parse_args()
    pre = {r["id"]: r for r in read_jsonl(ROOT / a.pre)}
    v1 = v1_spans()
    bot = next(u["id"] for u in call("/api/users/") if u.get("username") == BOT)
    while True:
        print(json.dumps(run(a, pre, v1, bot)), flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
