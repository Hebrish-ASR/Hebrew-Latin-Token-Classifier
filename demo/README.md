---
title: Hebrew-in-Latin Token Classifier
emoji: 🔤
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
models:
  - danielrosehill/hebrew-latin-token-classifier
  - malper/taatiknet
datasets:
  - danielrosehill/hebrew-latin-code-switching
short_description: Find romanized Hebrew words inside English text
---

Demo of [`danielrosehill/hebrew-latin-token-classifier`](https://huggingface.co/danielrosehill/hebrew-latin-token-classifier),
which finds Hebrew words written in Latin letters inside English text, and of what it
is for: making text-to-speech pronounce those words as Hebrew.

Source, corpus pipeline and docs: <https://github.com/danielrosehill/Hebrew-Latin-Token-Classifier>.
The Space is published from that repo's `demo/` folder.
