---
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
| `train` | 3802 | 987 | 383 |
| `validation` | 467 | 124 | 37 |
| `test` | 468 | 124 | 38 |

## Format

```json
{
 "id": "g00000",
 "text": "Are you coming to our barbecue for yom haatzmaut this year?",
 "source": "generated",
 "spans": [
  {
   "start": 35,
   "end": 48,
   "surface": "yom haatzmaut",
   "term": "yom haatzmaut"
  }
 ],
 "masked": [],
 "tokens": [
  [
   "Are",
   "O"
  ],
  [
   "you",
   "O"
  ],
  [
   "coming",
   "O"
  ],
  [
   "to",
   "O"
  ],
  [
   "our",
   "O"
  ],
  [
   "barbecue",
   "O"
  ],
  [
   "for",
   "O"
  ],
  [
   "yom",
   "B-HE"
  ],
  [
   "...",
   "..."
  ]
 ]
}
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

- **generator**: see repo
- **annotator**: see repo
- **generated sentences**: 4279
- **model agreement**: 87.7%
- **human-reviewed tasks**: 757
- **built**: 2026-10-15

Generation was term-seeded: the model was given one Hebrew term and asked for
sentences using it, and any generation where the term did not word-boundary match was
rejected. Annotation was a second, **blind** pass by a different model family that was
not told the seed term. Agreements were auto-accepted, disagreements were reviewed by
a human, and a random 10% sample of the agreements was also human-audited.

### Dictated slice

458 sentences come from
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
