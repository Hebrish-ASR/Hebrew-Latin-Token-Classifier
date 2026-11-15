"""Link a span's surface form to its canonical term (label scheme v2.1, `term` field).

The term-disjoint splits need every span tied to a term. Lookup keys are orthography.normalise() of:
data/terms.csv `term` and its `variants`, and data/exclusions.csv `term` (anglicised words, place names).
Two canonical terms that collide on one key make that key unusable (it is dropped, not guessed).
A miss returns (surface.lower(), False): the span becomes its own term and is flagged as unmatched.

Manual links (data/term_links.csv, Daniel 2026-11-15): form -> lemma, with relation
    spelling   same word, different romanization (chanukah -> hanukkah)
    reading    same word, different pronunciation tradition (shabbos -> shabbat; hebrew_variant differs)
    inflection plural / construct / possessive (mitzvot -> mitzvah)
    derived    a derived word (kibbutznik -> kibbutz)
status: proposed | confirmed. Links override the automatic index, so they also resolve collisions.
lemma() gives the group a term belongs to; term-disjoint splits should use the lemma, so that
Shabbos and Shabbat cannot end up on both sides of train/test.
"""
from __future__ import annotations

import csv, functools, pathlib

from .orthography import normalise

ROOT = pathlib.Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=1)
def _index() -> dict[str, str]:
    pairs: list[tuple[str, str]] = []
    for r in csv.DictReader((ROOT / "data" / "terms.csv").open()):
        t = r["term"].strip().lower()
        pairs.append((t, t))
        for v in filter(None, (r.get("variants") or "").split("|")):
            pairs.append((v.strip().lower(), t))
    for r in csv.DictReader((ROOT / "data" / "exclusions.csv").open()):
        t = r["term"].strip().lower()
        pairs.append((t, t))
    idx: dict[str, str] = {}
    clash: set[str] = set()
    for form, term in pairs:
        k = normalise(form)
        if k in idx and idx[k] != term:
            clash.add(k)
        idx.setdefault(k, term)
    for k in clash:
        idx.pop(k, None)
    for form in _links():
        idx[normalise(form)] = lemma(form)   # a collided key resolves to the group's lemma
    return idx


@functools.lru_cache(maxsize=1)
def _exact() -> set[str]:
    terms = {r["term"].strip().lower() for r in csv.DictReader((ROOT / "data" / "terms.csv").open())}
    terms |= {r["term"].strip().lower() for r in csv.DictReader((ROOT / "data" / "exclusions.csv").open())}
    return terms | set(_links())


@functools.lru_cache(maxsize=1)
def _links() -> dict[str, dict]:
    p = ROOT / "data" / "term_links.csv"
    if not p.exists():
        return {}
    return {r["form"].strip().lower(): r for r in csv.DictReader(p.open())}


def lemma(term: str) -> str:
    seen, t = set(), term.strip().lower()
    while t in _links() and t not in seen:  # follow chains, guard cycles
        seen.add(t); t = _links()[t]["lemma"].strip().lower()
    return t


def link(surface: str) -> tuple[str, bool]:
    """Exact spelling first (so "Chanukah" stays its own term, grouped by lemma), then the normalised index."""
    if surface.strip().lower() in _exact():
        return surface.strip().lower(), True
    t = _index().get(normalise(surface))
    return (t, True) if t else (surface.strip().lower(), False)
