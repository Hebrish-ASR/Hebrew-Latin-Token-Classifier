"""Hugging Face Space: demo of the Hebrew-in-Latin token classifier.

Pass 1 (detect) is the published model. Pass 2 (Latin -> Hebrew script) is not built
yet; TaatikNet is offered as an opt-in stand-in so pass 3 (per-TTS-engine rendering)
can be shown end to end. See docs/pipeline.md in the GitHub repo.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from html import escape

import gradio as gr

from predict import HebrewSpans

MODEL = "danielrosehill/hebrew-latin-token-classifier"
TRANSLIT = "malper/taatiknet"
REPO = "https://github.com/danielrosehill/Hebrew-Latin-Token-Classifier"
NIQQUD = re.compile(r"[֑-ׇ]")

finder = HebrewSpans(MODEL)


@lru_cache(maxsize=1)
def transliterator():
    # Loaded on first use so the Space starts on the classifier alone.
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    return (AutoTokenizer.from_pretrained(TRANSLIT),
            AutoModelForSeq2SeqLM.from_pretrained(TRANSLIT).eval())


@lru_cache(maxsize=2048)
def to_hebrew(word: str) -> str:
    # TaatikNet is word-level: fed "Bituach Leumi" it fuses the two words into one,
    # so multi-word spans are converted word by word and re-joined.
    tok, model = transliterator()
    ids = model.generate(**tok(word.lower(), return_tensors="pt"),
                         num_beams=5, max_new_tokens=48)
    return NIQQUD.sub("", tok.decode(ids[0], skip_special_tokens=True)).strip()


def chatterbox_segments(text: str, spans: list[dict], key: str) -> list[list[str]]:
    segs, cursor = [], 0
    for s in spans:
        if s["start"] > cursor:
            segs.append([text[cursor:s["start"]], "en"])
        segs.append([s[key], "he"])
        cursor = s["end"]
    if cursor < len(text):
        segs.append([text[cursor:], "en"])
    return segs


def ssml(text: str, spans: list[dict], key: str) -> str:
    out, cursor = [], 0
    for s in spans:
        out.append(escape(text[cursor:s["start"]], quote=False))
        out.append(f'<lang xml:lang="he-IL">{escape(s[key], quote=False)}</lang>')
        cursor = s["end"]
    out.append(escape(text[cursor:], quote=False))
    return '<speak version="1.0" xml:lang="en-US">' + "".join(out) + "</speak>"


def run(text: str, threshold: float, convert: bool):
    text = (text or "").strip()
    if not text:
        return [], "[]", "[]", "", "Type a sentence first."
    spans = finder(text, threshold=threshold)
    key = "text"
    if convert and spans:
        for s in spans:
            s["hebrew"] = " ".join(to_hebrew(w) for w in s["text"].split())
        key = "hebrew"

    highlighted, cursor = [], 0
    for s in spans:
        if s["start"] > cursor:
            highlighted.append((text[cursor:s["start"]], None))
        label = f"HE {s['score']:.2f}" + (f" · {s['hebrew']}" if "hebrew" in s else "")
        highlighted.append((s["text"], label))
        cursor = s["end"]
    if cursor < len(text):
        highlighted.append((text[cursor:], None))

    segs = chatterbox_segments(text, spans, key)
    if not spans:
        note = ("No Hebrew spans found. That is the intended answer for plain English, "
                "and for words English already says acceptably (*Shabbat*, *hummus*, "
                "*kosher*), which are deliberately not flagged.")
    else:
        note = f"**{len(spans)} span{'s' * (len(spans) != 1)}** found."
        if convert:
            note += (" Hebrew script comes from TaatikNet, which is experimental here: "
                     "it is a generic transliterator and does not know spelling "
                     "conventions, so *makolet* may come back as מקולת instead of "
                     "מכולת. The planned pass 2 puts a curated lexicon in front of it.")
        else:
            note += (" Pass 2 is off, so the TTS outputs below still carry the Latin "
                     "text. Tick **Convert spans to Hebrew script** to fill it in.")
    return (highlighted, json.dumps(spans, ensure_ascii=False, indent=2),
            "[\n" + ",\n".join("  " + json.dumps(x, ensure_ascii=False) for x in segs) + "\n]", ssml(text, spans, key), note)


EXAMPLES = [
    ["I need to go to Bituach Leumi before the makolet closes."],
    ["Can you pick up some bamba from the makolet on the way home?"],
    ["The bank wants three more documents before they'll approve the mashkanta."],
    ["Bring your teudat zehut, they won't let you into the misrad without it."],
    ["We're already late, so yalla, get in the car."],
    ["My kupat holim finally approved the referral."],
    ["She's been keeping a detailed heshbon ishi since she started her new job."],
    ["We had hummus and challah before Shabbat dinner."],
    ["I'll meet you at the gas station near the highway."],
]

INTRO = f"""
# Hebrew-in-Latin token classifier

Finds **Hebrew words written in Latin letters** inside English text, the way English
speakers in Israel write and talk: *"pick up some **bamba** from the **makolet**"*.

Before this model there was nothing for it. GlotLID, fastText `lid.176` and NLLB
`lid218e` have no `heb_Latn` label, and no token-level code-switching model covers
Hebrew. Model: [`{MODEL}`](https://huggingface.co/{MODEL}) ·
Corpus: [`hebrew-latin-code-switching`](https://huggingface.co/datasets/danielrosehill/hebrew-latin-code-switching) ·
Code: [GitHub]({REPO})
"""

USE_CASES = """
## Use cases

### 1. Text-to-speech that pronounces Hebrew as Hebrew (the reason this exists)

English TTS reads romanized Hebrew as English. *makolet* comes out as "MACK-oh-let",
*mashkanta* gets English stress, and the guttural in *chatzer* becomes the "ch" in
*church*. No voice fixes this alone, because nothing tells it the word is Hebrew.

The pipeline has three passes. **Pass 1, detect**, is this model. **Pass 2, convert**
the span to Hebrew script, uses a curated lexicon with TaatikNet as fallback; in this
demo it is TaatikNet alone, as an opt-in. **Pass 3, render**, writes the result in
whatever form the TTS engine accepts:

| Engine | How the language switch is expressed | Output tab above |
| --- | --- | --- |
| [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) | `language_id` is per call, with no inline switch, so the line is split into segments and the audio is concatenated | **Chatterbox segments** |
| SSML engines: Azure, Google, Amazon Polly | Inline `<lang xml:lang="he-IL">` | **SSML** |
| No language control | Respell the word, or leave it in English | — |

The first user is [My Weird Prompts](https://myweirdprompts.com), an AI-generated
podcast produced in Jerusalem and voiced by Chatterbox.

### 2. Dictation and ASR post-processing

Speech recognisers trained on English turn *makolet* into "macaullete". Spans
detected in a draft transcript tell you where to apply a Hebrew-aware correction
lexicon, and give you a list of terms to bias the next decode towards.

### 3. Search, tagging and normalisation

Romanized Hebrew has no standard spelling: *chatzer* / *hatzer* / *khatser*. Detecting
the spans first lets you normalise them, or map them to Hebrew script, before
indexing. A search for one spelling then finds the others.

### 4. Subtitle and content QA

Flag lines that contain Hebrew terms for review by a bilingual editor, or check that a
glossary is applied consistently across an episode or a document set.

### 5. Beyond Hebrew

Nothing in the method is specific to Hebrew except the term list. The same recipe
works for any immigrant or heritage-language community whose words appear romanized
inside a majority language, such as Arabic, Russian or Amharic in English. The recipe:
term-seeded generation, blind annotation by a second model family, human review of
disagreements only, term-disjoint splits, and a small multilingual encoder.
"""

ABOUT = """
## How good is it

`xlm-roberta-base` fine-tuned on 4,737 sentences, 458 of them dictated. It is scored on
a **term-disjoint** test split: no Hebrew term in the test set appears in training, so
the numbers measure generalisation to words the model has never seen.

| Test slice | Sentences | Precision | Recall | Span F1 |
| --- | --- | --- | --- | --- |
| All | 468 | 0.928 | 0.953 | **0.941** |
| Dictated | 38 | 0.929 | 1.000 | 0.963 |

None of the 58 designed hard-negative sentences was flagged.

**Labelled by pronunciation, not etymology.** Words that English already says
acceptably are deliberately not flagged: *Shabbat*, *hummus*, *challah*, *kosher*,
*aliyah*, and city names. The posture is conservative because, in a TTS pipeline, a
false positive costs a sentence split and an extra synthesis call and still makes the
output worse, while a miss costs nothing.

**Threshold.** A word is tagged Hebrew when P(B-HE) + P(I-HE) is at least the
threshold. Raise it for fewer false positives.

**Known false positives.** Some place names (*Dizengoff*, *Ramat Gan*) are flagged
although the policy excludes them.

## Use it

```python
# predict.py is in the model repo and the GitHub repo
from predict import HebrewSpans
find = HebrewSpans()
find("I need to go to Bituach Leumi before the makolet closes.")
# [{'start': 16, 'end': 29, 'text': 'Bituach Leumi', 'score': ...},
#  {'start': 41, 'end': 48, 'text': 'makolet', 'score': ...}]
```

Use `predict.py` or `pipeline(..., aggregation_strategy="first")`. The default
`"simple"` aggregation cuts words short (*mako*), because only the first sub-token of
each word is trained.
"""

with gr.Blocks(title="Hebrew-in-Latin token classifier") as demo:
    gr.Markdown(INTRO)
    with gr.Accordion("Why this model exists, in the author's voice (6½ min)", open=False):
        gr.Audio("why-this-model.mp3", label="Author's voice note", type="filepath",
                 interactive=False)
        gr.Markdown(f"Transcript: [`docs/voice-note/transcript.md`]"
                    f"({REPO}/blob/main/docs/voice-note/transcript.md)")
    with gr.Row():
        with gr.Column(scale=3):
            inp = gr.Textbox(label="English text", lines=3,
                             value=EXAMPLES[0][0],
                             placeholder="Type English with some romanized Hebrew in it")
        with gr.Column(scale=1):
            threshold = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="Threshold")
            convert = gr.Checkbox(label="Convert spans to Hebrew script (TaatikNet, experimental)",
                                  value=False)
            btn = gr.Button("Detect", variant="primary")
    note = gr.Markdown()
    with gr.Tabs():
        with gr.Tab("Highlighted"):
            out_hl = gr.HighlightedText(label="Detected spans", combine_adjacent=False,
                                        show_legend=False)
        with gr.Tab("Spans (JSON)"):
            out_json = gr.Code(language="json", label="Pass 1 output (+ pass 2 if enabled)")
        with gr.Tab("Chatterbox segments"):
            out_segs = gr.Code(language="json",
                               label="(text, language_id) pairs, one generate() call each")
        with gr.Tab("SSML"):
            out_ssml = gr.Code(language="html", label="SSML for Azure / Google / Polly")
    gr.Examples(EXAMPLES, inputs=inp)
    gr.Markdown(USE_CASES)
    gr.Markdown(ABOUT)

    outs = [out_hl, out_json, out_segs, out_ssml, note]
    btn.click(run, [inp, threshold, convert], outs)
    inp.submit(run, [inp, threshold, convert], outs)
    demo.load(run, [inp, threshold, convert], outs)

if __name__ == "__main__":
    demo.launch()
