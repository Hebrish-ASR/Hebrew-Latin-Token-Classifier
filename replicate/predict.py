"""Replicate predictor. The span logic is HebrewSpans from the predict.py shipped in
the HF model repo (identical to training/predict.py), loaded from /weights so the
image and the weights are the same pinned revision."""
import importlib.util

from cog import BasePredictor, Input

# Loaded under another name: cog imports this file as `predict`, so a plain
# `from predict import ...` would resolve to this file, not the HF copy.
_spec = importlib.util.spec_from_file_location("hf_predict", "/weights/predict.py")
_hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hf)
HebrewSpans = _hf.HebrewSpans


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.finder = HebrewSpans("/weights", device="cpu")

    def predict(
        self,
        text: str = Input(description="English text that may contain Hebrew words in Latin letters."),
        threshold: float = Input(
            description="Minimum P(Hebrew) for a word. Raise it for fewer false positives (TTS).",
            default=0.5, ge=0.0, le=1.0,
        ),
    ) -> list[dict]:
        return self.finder(text, threshold=threshold)
