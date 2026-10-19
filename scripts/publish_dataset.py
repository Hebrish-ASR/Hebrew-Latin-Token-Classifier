#!/usr/bin/env python3
"""Stage 6 — push the corpus to Hugging Face.

Uploads data/corpus/*.jsonl plus the term inventory and a dataset card. Reads
HUGGINGFACE_TOKEN from the environment (needs `write` role).

Dry run by default. Nothing is uploaded until --push is passed, because this
publishes to a public URL.

    python3 scripts/publish_dataset.py                    # show what would happen
    python3 scripts/publish_dataset.py --push
    python3 scripts/publish_dataset.py --push --private
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
TERMS = ROOT / "data" / "terms.csv"

CARD = """---
license: mit
language:
  - en
  - he
task_categories:
  - token-classification
tags:
  - code-switching
  - hebrew
  - transliteration
  - romanization
  - named-entity-recognition
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
      - split: test
        path: data/test.jsonl
---

# Hebrew-in-Latin code-switching corpus

English sentences containing **Hebrew words written in Latin characters**, with
token-level BIO spans marking them.

*"I'm going to **Bituach Leumi** today"* · *"pick up some **bamba** from the
**makolet**"*

No corpus like this was published when this was built. There is no `heb_Latn` label
in GlotLID v3, `lid.176` or `lid218e`, no token-level code-switching model covering
Hebrew, and no Hebrew-English code-switching corpus of any kind.

## What it is for

Training a classifier that finds Hebrew words in English text so a text-to-speech
system can pronounce them as Hebrew. The labelling test is **pronunciation, not
etymology**: would an English voice mispronounce this word, and would treating it as
Hebrew fix that? So *kosher*, *Shabbat* and *hummus* are negatives — English says them
acceptably — while *makolet*, *mashkanta* and *arnona* are positives.

## Why: text-to-speech that stops botching Hebrew

English TTS reads romanized Hebrew as English: *makolet* comes out as "MACK-oh-let"
and *mashkanta* gets English stress. The fix is to find the Hebrew spans, convert them
to Hebrew script, and send each one to the TTS engine with the **Hebrew language
identifier**. For Chatterbox that is a separate `language_id="he"` call; for SSML
engines it is `<lang xml:lang="he-IL">`. The rest of the sentence stays English.

The first user is [**My Weird Prompts**](https://myweirdprompts.com), an AI-generated
podcast produced in Jerusalem whose episodes often contain Hebrew words that have long
been mispronounced. The approach carries over to other immigrant and heritage-language
communities that mix their own words into a majority language; only the term list is
Hebrew-specific.

**Related work.** Hebrew-script NLP and speech are well covered by
[ivrit.ai](https://huggingface.co/ivrit-ai), [DICTA](https://huggingface.co/dicta-il),
[Phonikud](https://github.com/thewh1teagle/phonikud) and
[TaatikNet](https://huggingface.co/malper/taatiknet). The opposite direction,
foreign words written in Hebrew script, was studied by Goldberg & Elhadad (2008). No
language-ID system has a `heb_Latn` label, and before this there was no model for
Hebrew written in Latin letters inside English.

## Splits are term-disjoint

**No seed term appears in more than one split.** Splitting by sentence rather than by
term would let a model memorise a term in training and be scored on it at test time.
Evaluate generalisation to unseen terms, which is what this is for.

| Split | Sentences | Seed terms | Dictated sentences |
| --- | --- | --- | --- |
{split_table}

## Format

```json
{example}
```

`tokens` is `[token, tag]` pairs, tags in `O` / `B-HE` / `I-HE`. Multi-token terms
such as *bituach leumi* are `B-HE I-HE`, which is why the scheme is not a binary flag.

`masked` lists character spans that carry **no label**: spans the two annotating
models disagreed on and no human has ruled on yet, and Israeli person names, on which
the annotation policy has not ruled. Their tokens are tagged `O` in `tokens` for
format compatibility, so **ignore tokens inside a `masked` span** when training or
scoring (label `-100`). Tagging them `O` teaches a model that Hebrew words are English.

`source` is `generated` or `dictated`.

## How it was built

{provenance}

Generation was term-seeded: the model was given one Hebrew term and asked for
sentences using it, and any generation where the term did not word-boundary match was
rejected. Annotation was a second, **blind** pass by a different model family that was
not told the seed term. Agreements were auto-accepted, disagreements were reviewed by
a human, and a random {audit_pct}% sample of the agreements was also human-audited.

### Dictated slice

{dictated} sentences come from
[`danielrosehill/English-Hebrew-Codeswitching-Sentences`](https://huggingface.co/datasets/danielrosehill/English-Hebrew-Codeswitching-Sentences):
sentences the author read aloud on his phone for a speech-recognition fine-tune, so a
human chose to say every one of them. Only the text is used here. Labels are
recomputed by word-boundary match against the term inventory (exclusions applied),
not copied from the recording metadata. A sentence with no Hebrew span and no
known-English target term is dropped, since that is a lexicon gap rather than a
negative. Each sentence goes into the split that already holds its terms, so the
splits stay term-disjoint.

**The rest of this corpus is synthetic.** Sentences were model-generated and
model-annotated, with human review concentrated on disagreements and a random audit
rather than spread over every row. Treat the audit result as the quality estimate.

## Provenance and licence

MIT. Built in
[Hebrish-ASR/Hebrew-Latin-Token-Classifier](https://github.com/Hebrish-ASR/Hebrew-Latin-Token-Classifier),
which holds the generation, annotation and review code, and the annotation policy.

Model trained on it: [`danielrosehill/hebrew-latin-token-classifier`](https://huggingface.co/danielrosehill/hebrew-latin-token-classifier).

Related: [`Hebrish/English-Hebrew-Mixed-Sentences`](https://huggingface.co/datasets/Hebrish/English-Hebrew-Mixed-Sentences)
(human-read audio, the predecessor) and
[`danielrosehill/Whisper-Hebrish`](https://huggingface.co/danielrosehill/Whisper-Hebrish).
"""


def build_card(agreement: dict) -> str:
    rows, example = [], {}
    for split in ("train", "validation", "test"):
        path = CORPUS / f"{split}.jsonl"
        if not path.exists():
            continue
        items = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rows.append(f"| `{split}` | {len(items)} | "
                    f"{len({i['seed_term'] for i in items if i['seed_term']})} | "
                    f"{sum(1 for i in items if i.get('source') == 'dictated')} |")
        if not example and items:
            first = next((i for i in items if i["spans"]), items[0])
            example = {k: first[k] for k in ("id", "text", "source", "spans", "masked", "tokens")}
            example["tokens"] = example["tokens"][:8] + [["...", "..."]]

    provenance = "\n".join(
        f"- **{k}**: {v}" for k, v in {
            "generator": agreement.get("generator", "see repo"),
            "annotator": agreement.get("annotator", "see repo"),
            "generated sentences": agreement.get("sentences", "—"),
            "model agreement": f"{100 * (1 - agreement.get('disagreement_rate', 0)):.1f}%",
            "human-reviewed tasks": agreement.get("review_tasks", "—"),
            "built": datetime.date.today().isoformat(),
        }.items())

    report = json.loads((CORPUS / "report.json").read_text())
    return CARD.format(
        dictated=report.get("dictated_sentences", 0),
        split_table="\n".join(rows) or "| — | — | — |",
        example=json.dumps(example, ensure_ascii=False, indent=1),
        provenance=provenance,
        audit_pct=int(100 * agreement.get("audit_rate", 0.10)),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="Hebrish/hebrew-latin-code-switching")
    p.add_argument("--private", action="store_true")
    p.add_argument("--push", action="store_true", help="actually upload")
    args = p.parse_args()

    if not (CORPUS / "train.jsonl").exists():
        raise SystemExit("no corpus; run scripts/build_splits.py")

    agreement_path = ROOT / "data" / "generated" / "agreement.json"
    agreement = json.loads(agreement_path.read_text()) if agreement_path.exists() else {}
    card = build_card(agreement)
    (CORPUS / "README.md").write_text(card)

    files = sorted(CORPUS.glob("*.jsonl")) + [CORPUS / "README.md"]
    if TERMS.exists():
        files.append(TERMS)

    print(f"repo      {args.repo} ({'private' if args.private else 'public'})")
    print(f"card      {(CORPUS / 'README.md').relative_to(ROOT)}")
    for f in files:
        print(f"  upload  {f.relative_to(ROOT)}  ({f.stat().st_size:,} bytes)")

    if not args.push:
        print("\ndry run — pass --push to upload. This publishes to a public URL.")
        return

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit("HUGGINGFACE_TOKEN not set")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("pip install huggingface_hub")

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    for f in files:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"data/{f.name}" if f.suffix == ".jsonl" else f.name,
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"  pushed  {f.name}")
    print(f"\nhttps://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
