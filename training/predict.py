# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "transformers>=4.46,<5"]
# ///
"""Find Hebrew-in-Latin spans in English text with the published classifier.

    uv run training/predict.py "I need to go to Bituach Leumi before the makolet closes."

Words are pre-split with the same regex the corpus was tokenised with, and each word
takes the label of its first sub-token, which is the only one trained. This is why
it is used instead of the transformers `pipeline`. The pipeline splits on whitespace,
so trailing punctuation sticks to spans ("makolet?"), and the "simple" aggregation
cuts words short ("mako").
"""
from __future__ import annotations

import re
import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL = "danielrosehill/hebrew-latin-token-classifier"
TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")  # = scripts/lib/corpus.py


class HebrewSpans:
    def __init__(self, model: str = MODEL, device: str | None = None):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(model).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, text: str, threshold: float = 0.5) -> list[dict]:
        words = list(TOKEN_RE.finditer(text))
        if not words:
            return []
        enc = self.tok([m.group(0) for m in words], is_split_into_words=True,
                       truncation=True, return_tensors="pt").to(self.device)
        probs = self.model(**enc).logits[0].softmax(-1).cpu()
        labels = self.model.config.id2label
        tags, seen = {}, set()
        for i, w in enumerate(enc.word_ids()):
            if w is None or w in seen:
                continue
            seen.add(w)
            p = probs[i]
            # A word counts as Hebrew when P(B)+P(I) clears the threshold; raise it
            # for fewer false positives, which is what a TTS pipeline usually wants.
            he = float(p[1] + p[2])
            tags[w] = (labels[int(p[1:].argmax()) + 1] if he >= threshold else "O", he)
        spans: list[dict] = []
        for w, m in enumerate(words):
            tag, score = tags.get(w, ("O", 0.0))
            if tag == "O":
                continue
            prev = spans[-1] if spans else None
            # I-HE extends a span only when it directly follows one; an orphan I-HE
            # starts its own span rather than being dropped.
            if prev and tag == "I-HE" and prev["_last"] == w - 1:
                prev.update(end=m.end(), _last=w, score=min(prev["score"], score))
            else:
                spans.append({"start": m.start(), "end": m.end(), "score": score, "_last": w})
        for s in spans:
            s.pop("_last")
            s["text"] = text[s["start"]:s["end"]]
            s["score"] = round(s["score"], 3)
        return spans


if __name__ == "__main__":
    finder = HebrewSpans()
    for line in sys.argv[1:] or sys.stdin:
        line = line.strip()
        print(line, "->", [(s["text"], s["score"]) for s in finder(line)])
