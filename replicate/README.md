# Replicate packaging

Live model: <https://replicate.com/danielrosehill/hebrew-latin-token-classifier>,
public, hardware `cpu`. First published 2026-10-16 as version
`660aa11ed7cb74aa2fb01aa6f6a2bee77f0e643487f13c02719158644cdeb5f6`, built from HF
revision `cb15ef8e30618bc255fd9a1fc35a128bf7741cc5`.

## Interface

| Input | Type | Default |
| --- | --- | --- |
| `text` | string | (required) |
| `threshold` | float 0–1 | `0.5`: minimum P(B-HE)+P(I-HE) for a word to count as Hebrew |

Output is a JSON list of spans, in the same shape `training/predict.py` returns:

```json
[{"start": 16, "end": 29, "score": 0.998, "text": "Bituach Leumi"},
 {"start": 41, "end": 48, "score": 0.999, "text": "makolet"}]
```

## How it is built

- The weights are **baked into the image** at build time (the `run:` step in `cog.yaml`),
  from a **pinned HF revision**, so a cold start never touches the Hub. Only the
  `*.json` files, `model.safetensors` and `predict.py` are fetched. The voice note,
  figures and `training_args.bin` are left out.
- **There is only one copy of the span logic.** `predict.py` here loads `HebrewSpans` from
  `/weights/predict.py`, the copy shipped in the HF repo, which is identical to
  `training/predict.py` (checked 2026-10-16). It is loaded with `importlib` under the name
  `hf_predict` because cog imports this file as `predict`. A plain
  `from predict import HebrewSpans` would import this file itself.
- PyTorch is installed from the CPU wheel index (`requirements.txt`), because the default
  wheel pulls in several GB of CUDA libraries a CPU model never uses.

## Updating after a retrain

1. Push the new weights to HF, then take the new revision from
   `curl -s https://huggingface.co/api/models/danielrosehill/hebrew-latin-token-classifier | jq -r .sha`.
2. Put it in `revision=` in `cog.yaml`. If `predict.py` on HF changed, check that
   `HebrewSpans(model, device)` and `__call__(text, threshold)` still have the same signatures.
3. Test locally: `cog predict -i text="I need to go to Bituach Leumi before the makolet closes."`
4. Push with `cog push`. This needs `docker login r8.im`. The token is the 1Password item
   "Claude Code Replicate Key (Personal)" (vault `Daniel - Personal`). It is a secure
   note, so the key is in the `text` field, and `op://…/notesPlain` comes back empty.
5. Check the hosted version with a real prediction (`POST /v1/predictions` with the
   `latest_version` id). A successful push only shows that the image uploaded.

## Known noise (harmless)

- cog 0.23 logs `BasePredictor`/`predict()` deprecation warnings. It prefers
  `BaseRunner`/`run()`, but the old API still works.
- transformers warns about "an incorrect regex pattern … fix_mistral_regex". This is a
  false positive on the XLM-R tokenizer: the spans it produces match the model card
  examples and the local `training/predict.py`.
