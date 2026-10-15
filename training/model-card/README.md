---
license: mit
language:
  - en
  - he
library_name: transformers
pipeline_tag: token-classification
base_model: FacebookAI/xlm-roberta-base
datasets:
  - danielrosehill/hebrew-latin-code-switching
tags:
  - code-switching
  - hebrew
  - transliteration
  - romanization
  - text-to-speech
metrics:
  - f1
  - precision
  - recall
model-index:
  - name: hebrew-latin-token-classifier
    results:
      - task:
          type: token-classification
        dataset:
          name: hebrew-latin-code-switching (test, term-disjoint)
          type: danielrosehill/hebrew-latin-code-switching
          split: test
        metrics:
          - {type: f1, value: 0.941, name: span F1}
          - {type: precision, value: 0.928, name: span precision}
          - {type: recall, value: 0.953, name: span recall}
---

# Hebrew-in-Latin token classifier

Finds **Hebrew words written in Latin letters** inside English text.

```
I need to go to Bituach Leumi before the makolet closes.
                [Bituach Leumi]          [makolet]
```

There was no model for this before. GlotLID, `lid.176` and `lid218e` have no
`heb_Latn` label, and no token-level code-switching model covers Hebrew. It was built
so a text-to-speech system can send these spans to a Hebrew voice instead of reading
*makolet* as English letters. The labelling test is **pronunciation, not etymology**.
Words English already says acceptably are deliberately **not** flagged: *Shabbat*,
*hummus*, *challah*, *kosher*, *aliyah*, and city names under the default posture.

## Results

`xlm-roberta-base`, 8 epochs, about 3 minutes on one A10G. Scored on the
**term-disjoint** test split: no Hebrew term in test appears in training, so this
measures how well the model generalises to words it has never seen.

| Test slice | Sentences | Span precision | Span recall | Span F1 |
| --- | --- | --- | --- | --- |
| All | 468 | 0.928 | 0.953 | **0.941** |
| Generated | 430 | 0.928 | 0.948 | 0.938 |
| Dictated (human-chosen sentences) | 38 | 0.929 | 1.000 | 0.963 |
| Validation (model selection) | 467 | 0.950 | 0.948 | 0.949 |

**Hard negatives: 0 false positives in 58.** These are sentences built to trap the
model: substring traps, the English sense of homographs (*dud*, *tor*, *salon*),
lookalikes, and anglicised loanwords. Of the 41 test sentences it gets wrong, the
false positives fall into two groups:

- **Real policy misses.** Place names the conservative policy excludes (*Dizengoff*,
  *Ramat Gan*, *Talpiot*), *mishnah*, *babaganoush*, and once *venv*.
- **Gold-label gaps.** Hebrew words the blind annotator refused that no human has yet
  re-reviewed: *msabbaha*, *lechem*, *sakin*, *kapsulot*, *od paam*. Here the model is
  arguably right.

Misses are mostly words that are also common Hebrew loanwords or short (*shakshuka*,
*sukkot*, *botz*, *chamutz*), and the third word of three-word compounds
(*heshbonit mas kabala*).

For comparison, the closest published recipe (EnTaCs, English–Tamil, arXiv 2603.26587)
reports F1 0.638 on its romanized class. The project planned for about that.

## Use

Pre-split words the way the training data was split, and take each word's label from
its **first sub-token**. This is the only one trained. `predict.py` in this repo does
both:

```python
# uv run predict.py "Grab some bamba from the makolet."   (or copy the class)
from predict import HebrewSpans
find = HebrewSpans()                       # danielrosehill/hebrew-latin-token-classifier
find("My savta says the arnona went up again.")
# [{'start': 3, 'end': 8, 'text': 'savta', 'score': 1.0},
#  {'start': 18, 'end': 24, 'text': 'arnona', 'score': 1.0}]
find("...", threshold=0.8)                 # fewer false positives, fewer hits
```

The stock `pipeline` also works, with `aggregation_strategy="first"`:

```python
from transformers import pipeline
ner = pipeline("token-classification", model="danielrosehill/hebrew-latin-token-classifier",
               aggregation_strategy="first")
```

It splits words on whitespace, though, so trailing punctuation attaches to a span
(`makolet?`). **Do not use `aggregation_strategy="simple"`**: it cuts words short
(`mako`).

Labels: `O`, `B-HE`, `I-HE`. Multi-word terms come out as one span (*Bituach Leumi*,
*kupat holim*).

## Training data

[`danielrosehill/hebrew-latin-code-switching`](https://huggingface.co/datasets/danielrosehill/hebrew-latin-code-switching):
4,737 sentences over 1,235 seed terms.

- **Generated sentences.** Each one was seeded with a term, then annotated blind by a
  second model family. Disagreements went to human review, along with a 10% random
  audit of agreements.
- **458 dictated sentences.** These are the text of sentences the author read aloud
  for a speech-recognition fine-tune.
- **Masked spans.** Spans nobody has ruled on, and Israeli person names, carry label
  `-100`: no loss, and not scored.

The data is largely synthetic. Treat the dictated slice as the closest thing to real
input.

## Limits

- It is English text with romanized Hebrew. Hebrew script is not the target and is
  not handled.
- Romanization varies. Common spelling variants were trained on, but an unusual
  spelling of an unseen word is the hardest case.
- **Person names are unlabelled** (masked in training), so their behaviour is
  undefined. **Place names** are negative by policy, but the model sometimes flags
  Hebrew-sounding ones.
- **Precision matters most** downstream: a false positive gets an English word read
  in Hebrew. Raise `threshold` if that matters more than recall.

## Provenance

Code, annotation policy and taxonomy:
[danielrosehill/Hebrew-Latin-Token-Classifier](https://github.com/danielrosehill/Hebrew-Latin-Token-Classifier).
The training script is `training/train.py`, and `eval_results.json` and
`test_errors.jsonl` in this repo hold the full test output. Trained 2026-10-15. MIT.
