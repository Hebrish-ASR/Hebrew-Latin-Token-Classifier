#!/usr/bin/env python3
"""Stage 5a — import the dictated code-switching sentences as a labelled slice.

Source: danielrosehill/English-Hebrew-Codeswitching-Sentences, sentences Daniel read
aloud on his phone for the Parakeet fine-tune
(~/repos/github/audio-dictation/ASR-Training). Only the TEXT is used
here. These are the only sentences in the corpus a human chose to say, so they double
as the closest thing to real input this classifier has.

Their `target_terms` were matched by word boundary against this repo's lexicon, but
they also include people, places and tech terms, so labels are recomputed here rather
than trusted:

  lexicon term (active, not excluded) found in text   -> span (B-HE / I-HE)
  `ambiguous` lexicon term                            -> span only if the sentence
                                                         was written for it (target_terms)
  exclusions.csv term, place name, tech term          -> O, same as the generated corpus
  person name, or a target term nobody has ruled on   -> MASKED: no label, no loss

Person names are masked rather than tagged O because the taxonomy has not ruled on
them (2026-10-15), and an O on "Osnat" would teach that Hebrew-looking words are
English, which is the opposite of the point.

    python3 scripts/fetch_dictated.py [--names PATH --tech PATH]

Output: data/dictated/sentences.jsonl, read by build_splits.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import corpus, orthography  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "danielrosehill/English-Hebrew-Codeswitching-Sentences"
PARAKEET = pathlib.Path.home() / "repos/github/audio-dictation/ASR-Training/data"
OUT = ROOT / "data" / "dictated"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--names", type=pathlib.Path, default=PARAKEET / "israeli-names.json")
    p.add_argument("--tech", type=pathlib.Path, default=PARAKEET / "tech-terms.json")
    p.add_argument("--revision", default="main")
    args = p.parse_args()

    terms = {r["term"].lower(): r for r in csv.DictReader(open(ROOT / "data/terms.csv"))}
    excluded = {r["term"].lower() for r in csv.DictReader(open(ROOT / "data/exclusions.csv"))}
    names = json.loads(args.names.read_text())["names"]
    people = {v.lower() for n in names if n["kind"] != "place"
              for v in [n["name"], *n["variants"]]}
    places = {v.lower() for n in names if n["kind"] == "place"
              for v in [n["name"], *n["variants"]]}
    tech = {t["term"].lower() for t in json.loads(args.tech.read_text())["terms"]}

    # lexicon.csv holds terms recovered from English-Hebrew-Mixed-Sentences that the
    # generation inventory never re-seeded (chufshat leida, pensia, Ynet, Janglo...).
    # Many dictated prompts came from that dataset, so without them those sentences
    # would carry real Hebrew labelled O.
    legacy = {r["term"].lower() for r in csv.DictReader(open(ROOT / "data/lexicon.csv"))
              if r["status"] == "active"}
    active = [t for t in {*(t for t, r in terms.items() if r["status"] == "active"), *legacy}
              if t not in excluded]
    variant_of = orthography.variant_map(terms)
    # Longest first, so "bituach leumi" claims its words before "bituach" does.
    matchable = sorted({*active, *(v for v, t in variant_of.items() if t in active)},
                       key=len, reverse=True)

    rows, stats = [], {"masked_person": 0, "masked_unruled": 0, "o_known": 0,
                       "dropped_unexplained_empty": 0}
    for split in ("train", "test"):
        path = hf_hub_download(REPO, f"data/{split}/metadata.jsonl", repo_type="dataset",
                               revision=args.revision)
        for r in map(json.loads, open(path)):
            text, spans, masked = r["text"], [], []

            def free(s: int, e: int) -> bool:
                return not any(s < x["end"] and e > x["start"] for x in spans + masked)

            targets = [t.lower() for t in r["target_terms"]]
            known_o = False
            for t in matchable + [t for t in targets
                                  if terms.get(t, {}).get("status") == "ambiguous"]:
                for s, e in corpus.word_spans(text, t):
                    if free(s, e):
                        spans.append({"start": s, "end": e, "surface": text[s:e],
                                      "term": variant_of.get(t, t)})
            for t in targets:
                if t in terms or t in excluded or t in places or t in tech:
                    stats["o_known"] += 1
                    known_o = known_o or t not in terms
                    continue
                kind = "masked_person" if all(w in people for w in t.split()) else "masked_unruled"
                for s, e in corpus.word_spans(text, t):
                    if free(s, e):
                        masked.append({"start": s, "end": e, "surface": text[s:e]})
                        stats[kind] += 1
            # Every row here was written as code-switching. One with no span and no
            # target that is a known negative almost certainly holds a Hebrew word the
            # lexicon lacks, so it is dropped rather than shipped as a false negative.
            if not spans and not masked and not known_o:
                stats["dropped_unexplained_empty"] += 1
                continue
            spans.sort(key=lambda x: x["start"])
            rows.append({"id": f"d_{r['id']}", "text": text, "source_split": split,
                         "spans": spans, "masked": sorted(masked, key=lambda x: x["start"]),
                         "recorded_at": r.get("recorded_at"), "source": REPO})

    OUT.mkdir(parents=True, exist_ok=True)
    corpus.write_jsonl(OUT / "sentences.jsonl", rows)
    stats.update(sentences=len(rows), with_spans=sum(1 for r in rows if r["spans"]),
                 spans=sum(len(r["spans"]) for r in rows))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
