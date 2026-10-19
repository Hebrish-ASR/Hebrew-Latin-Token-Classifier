# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "transformers>=4.46,<5",  # v5 removed TrainingArguments.warmup_ratio (2026-10-15)
#   "accelerate",
#   "datasets>=3",
#   "seqeval",
#   "huggingface_hub>=0.26",
# ]
# ///
"""Fine-tune a token classifier for Hebrew-in-Latin spans and push it to the Hub.

Runs anywhere; written for Hugging Face Jobs:

    hf jobs uv run --flavor a10g-small --secrets HF_TOKEN training/train.py \\
        --push danielrosehill/hebrew-latin-token-classifier

Data: Hebrish/hebrew-latin-code-switching (term-disjoint splits). Each row
carries `tokens` ([word, BIO tag]) and `masked` (character spans with no label:
unreviewed model disagreements and person names). Words inside a masked span get
label -100 -- no loss, not scored -- because tagging them O would teach the model
that Hebrew words are English.

Model selection is on validation span-F1. Test is scored once, at the end, overall
and separately on the dictated slice (sentences a human chose to say), and written to
eval_results.json in the pushed repo.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re

import numpy as np
from datasets import load_dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments)

# Must match scripts/lib/corpus.py TOKEN_RE -- the corpus `tokens` were cut with it.
TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")
LABELS = ["O", "B-HE", "I-HE"]
L2I = {l: i for i, l in enumerate(LABELS)}


def word_labels(row: dict) -> tuple[list[str], list[int]]:
    words, tags = zip(*row["tokens"]) if row["tokens"] else ((), ())
    offsets = [(m.start(), m.end()) for m in TOKEN_RE.finditer(row["text"])]
    if [row["text"][s:e] for s, e in offsets] != list(words):
        raise ValueError(f"tokenisation drift in {row['id']}")
    masked = row.get("masked") or []
    labels = []
    for (s, e), tag in zip(offsets, tags):
        hidden = any(s < m["end"] and e > m["start"] for m in masked)
        labels.append(-100 if hidden else L2I[tag])
    return list(words), labels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Hebrish/hebrew-latin-code-switching")
    p.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    p.add_argument("--epochs", type=float, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--out", default="out")
    p.add_argument("--push", help="model repo id to push to")
    args = p.parse_args()

    ds = load_dataset(args.dataset)
    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(row):
        words, labels = word_labels(row)
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=160)
        aligned, prev = [], None
        for w in enc.word_ids():
            # Label the first sub-token of each word only.
            aligned.append(-100 if w is None or w == prev else labels[w])
            prev = w
        enc["labels"] = aligned
        return enc

    keep = {"id", "source"}
    enc = ds.map(encode, remove_columns=[c for c in ds["train"].column_names if c not in keep])

    def to_tags(preds, labels):
        y_true, y_pred = [], []
        for pr, la in zip(preds, labels):
            t, q = [], []
            for a, b in zip(pr, la):
                if b != -100:
                    t.append(LABELS[b]); q.append(LABELS[a])
            y_true.append(t); y_pred.append(q)
        return y_true, y_pred

    def scores(y_true, y_pred):
        return {"precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred)}

    def compute_metrics(ev):
        return scores(*to_tags(np.argmax(ev.predictions, -1), ev.label_ids))

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=L2I)

    targs = TrainingArguments(
        output_dir=args.out, learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=64,
        warmup_ratio=0.1, weight_decay=0.01, lr_scheduler_type="linear",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        logging_steps=20, report_to="none", seed=args.seed,
        bf16=_bf16_ok(), fp16=not _bf16_ok() and _cuda())
    trainer = Trainer(model=model, args=targs,
                      train_dataset=enc["train"].remove_columns(["id", "source"]),
                      eval_dataset=enc["validation"].remove_columns(["id", "source"]),
                      processing_class=tok,
                      data_collator=DataCollatorForTokenClassification(tok),
                      compute_metrics=compute_metrics)
    trainer.train()

    results = {"model": args.model, "dataset": args.dataset,
               "hyperparameters": {k: getattr(args, k) for k in ("epochs", "lr", "batch", "seed")},
               "best_checkpoint": trainer.state.best_model_checkpoint,
               "validation": trainer.evaluate()}
    test = enc["test"]
    pred = trainer.predict(test.remove_columns(["id", "source"]))
    y_true, y_pred = to_tags(np.argmax(pred.predictions, -1), pred.label_ids)
    results["test"] = scores(y_true, y_pred) | {"sentences": len(y_true)}
    for src in ("generated", "dictated"):
        idx = [i for i, s in enumerate(test["source"]) if s == src]
        if idx:
            results[f"test_{src}"] = scores([y_true[i] for i in idx],
                                            [y_pred[i] for i in idx]) | {"sentences": len(idx)}
    # Sentences with no gold span: how often does the model flag something anyway?
    neg = [i for i, t in enumerate(y_true) if "B-HE" not in t]
    results["test_negative_sentences"] = {
        "sentences": len(neg),
        "false_positive_sentence_rate": sum("B-HE" in y_pred[i] or "I-HE" in y_pred[i]
                                            for i in neg) / max(1, len(neg))}
    results["test_report"] = classification_report(y_true, y_pred, output_dict=False)
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            words = [w for w, _ in ds["test"][i]["tokens"]]
            errors.append({"id": test["id"][i], "text": ds["test"][i]["text"],
                           "gold": [w for w, t in zip(words, y_true[i]) if t != "O"],
                           "pred": [w for w, t in zip(words, y_pred[i]) if t != "O"]})
    print(json.dumps({k: v for k, v in results.items() if k != "test_report"}, indent=1))
    print(results["test_report"])

    out = pathlib.Path(args.out) / "final"
    trainer.save_model(out)
    tok.save_pretrained(out)
    (out / "eval_results.json").write_text(json.dumps(results, indent=1) + "\n")
    (out / "test_errors.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in errors))

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        # Private until the numbers have been read; flipped public by hand.
        api.create_repo(args.push, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push,
                          commit_message=f"Train {args.model} on {args.dataset}: "
                                         f"test F1 {results['test']['f1']:.3f}")
        print(f"pushed https://huggingface.co/{args.push}")


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def _bf16_ok() -> bool:
    import torch
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


if __name__ == "__main__":
    main()
