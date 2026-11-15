"""Label scheme v2 pre-annotation: an LLM labels every sentence first, a human corrects in Label Studio.

Scheme (docs/annotation-policy.md, "Label scheme v2", decided 2026-11-15):
    span tag        HEB (romanized Hebrew not established in English) | SHARED (Hebrew/Yiddish also established in English)
    lang            hebrew (default) | yiddish | aramaic | arabic      -- language switched into, not origin
    hebrew_variant  standard (default) | ashkenazic | sephardic | yemenite        -- only when lang = hebrew
    place names     untagged

The LLM runs blind: it does not see the v1 labels. ls_build_v2.py compares the two afterwards, so places where
they disagree can be reviewed first.

    with-secrets uv run scripts/v2_preannotate.py --pilot 30 --model google/gemini-3.8-flash
    with-secrets uv run scripts/v2_preannotate.py --all --model ...       # bulk: needs Daniel's yes on model + estimate

Output: data/v2/preannotations.<model-slug>.jsonl, one row per sentence:
    {id, split, text, model, at, spans: [{start, end, surface, tag, lang, hebrew_variant}]}
"""
from __future__ import annotations

import argparse, asyncio, csv, datetime as dt, json, pathlib, random, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import llm  # noqa: E402
from lib.corpus import read_jsonl, append_jsonl, word_spans  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "v2"
BATCH = 10

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "i": {"type": "integer"},
            "spans": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "surface": {"type": "string"},
                    "tag": {"type": "string", "enum": ["HEB", "SHARED"]},
                    "lang": {"type": "string", "enum": ["hebrew", "yiddish", "aramaic", "arabic"]},
                    "hebrew_variant": {"type": "string", "enum": ["standard", "ashkenazic", "sephardic", "yemenite"]},
                },
                "required": ["surface", "tag", "lang", "hebrew_variant"], "additionalProperties": False}},
        },
        "required": ["i", "spans"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False,
}


def system_prompt() -> str:
    excl = list(csv.DictReader((ROOT / "data" / "exclusions.csv").open()))
    shared = sorted(r["term"] for r in excl if r["category"] == "anglicised")
    places = sorted(r["term"] for r in excl if r["category"] == "place_name")
    return f"""You label English sentences that may contain Hebrew (or Yiddish, Aramaic, Arabic-via-Hebrew) words written in Latin letters.

For each sentence, list every such word or multi-word expression as a span. Copy "surface" exactly as it appears in the sentence (same case and spelling).

tag:
- HEB: romanized Hebrew that is NOT established in English. An English reader would not know it and an English TTS voice would mispronounce it. Examples: makolet, mashkanta, misrad hapnim, tofes, mazgan, machsom, ulpan, sababa, yalla, achla.
- SHARED: a Hebrew or Yiddish word that IS also established in English (in English dictionaries, known to general English readers). The known list includes: {", ".join(shared)}.
- Do NOT tag place names (cities, neighbourhoods, streets, regions, malls), e.g. {", ".join(places[:20])}. Common nouns that only sound geographic are still tagged (kikar, rehov, tayelet). In "kikar rabin", the whole thing is a place name and is not tagged.
- Do NOT tag ordinary English words, even if they look like Hebrew (hi, lo, ma, mi, at, ken as English).
- Brand and company names that are Hebrew words are HEB (e.g. Bituach Leumi, Egged's service names, Tnuva).
- Span boundaries: a fixed multi-word expression is one span (misrad hapnim, bituach leumi, yom haatzmaut). Don't include English words or punctuation.

lang: the language the speaker is switching INTO, not the word's etymology.
- "hebrew" (default): anything used as part of Hebrew speech, including Arabic-origin Israeli slang (yalla, sababa, achla, mabsut) and Aramaic-origin words that live in Hebrew (davka, kal vachomer, stam).
- "yiddish": Yiddish words (schlep, kvetch, nebach, shvitz, davening, bubbe).
- "aramaic": only an Aramaic phrase quoted as Aramaic, e.g. a Talmudic expression (ta shma, teiku).
- "arabic": only Arabic that is not part of Israeli Hebrew usage.

hebrew_variant (when lang is not hebrew, output "standard"; it is ignored): the default is "standard" (modern Israeli). Mark the exception only when the SPELLING itself signals it:
- "ashkenazic": s/ss for tav (Shabbos, Succos, bris, emes, Shavuos), -os plural endings (seforim is ambiguous; mitzvos is ashkenazic), penultimate-stress spellings like "Yontif", "Toirah". This applies to SHARED words too: "Hanukkah" is standard, "Chanukah"/"Chanukkah" is ashkenazic.
- "sephardic": a spelling that marks a traditional Sephardi or Mizrahi pronunciation that differs from modern Israeli, e.g. ḥ/ʿ gutturals written distinctly ("Ḥanukka", "'Ivrit" with a marked ayin), "th" for an unpointed tav as in "Shabbath" or "Berith".
- "yemenite": a spelling that marks Yemenite pronunciation (e.g. "j" for gimel as in "Jimel", distinct waw/vav, "Shabbath").
If unsure, "standard".

Return {{"items": [{{"i": <index>, "spans": [...]}}, ...]}} with one item per input sentence, echoing each index. Empty spans list if none."""


def load_corpus() -> list[dict]:
    rows = []
    for split in ("train", "validation", "test"):
        for r in read_jsonl(ROOT / "data" / "corpus" / f"{split}.jsonl"):
            r["split"] = split
            rows.append(r)
    return rows


def pilot_sample(rows: list[dict], n: int, seed: int = 7) -> list[dict]:
    """Stratified, so the pilot shows every hard case: shared-word negatives, open review items, dictated, traps."""
    rng = random.Random(seed)
    queue = json.loads((ROOT / "review" / "queue.json").read_text())
    queue = queue if isinstance(queue, list) else queue.get("tasks", [])
    decided = json.loads((ROOT / "review" / "decisions.json").read_text())
    open_ids = {t["id"] for t in queue if t.get("kind") == "span" and t["key"] not in decided}
    buckets = [
        ([r for r in rows if r["id"] in open_ids], round(n * .25)),
        ([r for r in rows if r["category"] == "negative:anglicised"], round(n * .2)),
        ([r for r in rows if r["source"] == "dictated"], round(n * .15)),
        ([r for r in rows if r["category"] in ("negative:english_sense", "negative:substring_trap", "negative:lookalike")], round(n * .1)),
    ]
    picked, seen = [], set()
    for pool, k in buckets:
        for r in rng.sample(pool, min(k, len(pool))):
            if r["id"] not in seen:
                picked.append(r); seen.add(r["id"])
    rest = [r for r in rows if r["id"] not in seen and not r["category"].startswith("negative")]
    picked += rng.sample(rest, n - len(picked))
    return picked


def locate(text: str, span: dict) -> list[dict]:
    hits = word_spans(text, span["surface"])
    out = []
    for s, e in hits:
        lang = span["lang"]
        out.append({"start": s, "end": e, "surface": text[s:e], "tag": span["tag"], "lang": lang,
                    "hebrew_variant": span["hebrew_variant"] if lang == "hebrew" else None})
    return out


async def run(rows: list[dict], model: str, out: pathlib.Path) -> None:
    done = {r["id"] for r in read_jsonl(out)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} sentences, {len(todo)} to label, model {model} -> {out.relative_to(ROOT)}")
    system = system_prompt()
    unlocated = 0
    async with llm.Client(model, concurrency=8, stage="v2-preannotate") as client:
        async def one(batch):
            nonlocal unlocated
            user = "\n".join(f"{i}. {r['text']}" for i, r in enumerate(batch))
            res = await client.json_completion(system, user, SCHEMA, temperature=0.0)
            by_i = {it["i"]: it["spans"] for it in res.get("items", [])}
            now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows_out = []
            for i, r in enumerate(batch):
                if i not in by_i:
                    continue                       # dropped: picked up on the next (resumable) run
                spans, seen = [], set()
                for sp in by_i[i]:
                    found = locate(r["text"], sp)
                    if not found:
                        unlocated += 1
                    for f in found:
                        if (f["start"], f["end"]) not in seen:
                            seen.add((f["start"], f["end"])); spans.append(f)
                spans.sort(key=lambda s: s["start"])
                rows_out.append({"id": r["id"], "split": r["split"], "text": r["text"], "model": model, "at": now,
                                 "spans": spans})
            append_jsonl(out, rows_out)
        await asyncio.gather(*(one(todo[k:k + BATCH]) for k in range(0, len(todo), BATCH)))
    print(f"done; {unlocated} returned spans could not be found in their sentence (dropped)")


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", type=int, help="stratified sample of N sentences")
    g.add_argument("--all", action="store_true")
    p.add_argument("--model", required=True)
    a = p.parse_args()
    rows = load_corpus()
    rows = pilot_sample(rows, a.pilot) if a.pilot else rows
    slug = re.sub(r"[^a-z0-9]+", "-", a.model.lower()).strip("-")
    out = OUT_DIR / (f"pilot.{slug}.jsonl" if a.pilot else f"preannotations.{slug}.jsonl")
    asyncio.run(run(rows, a.model, out))


if __name__ == "__main__":
    main()
