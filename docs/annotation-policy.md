# Annotation policy

What counts as a positive span. Written 2026-09-22, before review starts, so the
decisions are consistent across a session and legible to whoever picks this up next.

Open questions are marked **UNDECIDED** — settle them in the first twenty review tasks,
then record the answer here. Decisions already taken are marked **DECIDED** with a date.

## The label

> **Dataset label scheme v2: DECIDED 2026-11-15 (Daniel), not yet implemented.** See
> [Label scheme v2](#label-scheme-v2-decided-2026-11-15) below. The text in this section and in
> *Anglicised Hebrew* describes v1, which the released model (2026-10-15) was trained on.

One entity type, `HE`: a **Hebrew word rendered in Latin characters**. BIO tagging,
so a multi-token term is `B-HE I-HE` (51 of 139 known terms are multi-token).

## The operational test

Not "is this word etymologically Hebrew?" but:

> **Would an English TTS voice mispronounce it, and would tagging it as Hebrew fix
> that?**

This is a pronunciation task, not an etymology task. It follows that:

- **Words the English corpus has absorbed are negatives.** *Kosher*, *Shabbat*,
  *hummus*, *rabbi*, *kibbutz*, *chutzpah* are read acceptably by an English voice.
  Tagging them would change working output for no gain.
- **Words the English corpus lacks are positives.** *Bituach Leumi*, *makolet*,
  *mashkanta*, *tlush*, *mazgan*, *arnona*, *dud*.
- **Guttural phonemes are always positives.** *Challah*, *chagim*, *machsom*,
  *hashmal* — English orthography cannot represent ח or כ, so an English voice
  cannot get them right.

The boundary between the first two is a judgement call and will not be perfectly
consistent. When genuinely unsure, **say no**.

### Why the default is no — DECIDED 2026-09-22

Daniel's rule: *"each time we flag a word as being Hebrew in Latin characters it's
going to trigger downstream actions — it's going to add complication. I'd rather it's
selective on the things that truly get botched in TTS."*

A flagged span is not a label, it is **work**. It splits the sentence, adds a separate
speech call for the fragment, and requires the audio to be stitched back together with
padding tuned by ear. So the two errors are not symmetric:

| Error | Cost |
| --- | --- |
| Missed a Hebrew word | Current behaviour, unchanged. The word is read as it is read today |
| Flagged an English word | It gets spoken in Hebrew, **and** the pipeline does all that work to make it worse |

A miss is free. A false positive is expensive twice over. Err towards no.

### Anglicised Hebrew — always negative, and applied automatically

A large class of Hebrew words, modern and ancient, can be assumed present in an
English corpus. Today's TTS renders them acceptably with no Hebrew language
instruction at all. **They are always negative** — *kosher*, *kashrut*, *Shabbat*,
*challah*, *hummus*, *tahini*, *falafel*, *Torah*, *seder*, *Hanukkah*, *Sukkot*,
*shiva*, *menorah*, *yarmulke*, *mitzvah*, *bar mitzvah*, *Knesset*, *shekel*,
*aliyah*.

This is not left to per-case judgement. The class is known and finite, so it lives in
[`data/exclusions.csv`](../data/exclusions.csv) and `adjudicate.py` drops those spans
before the review queue is built. Reviewing a 126-line file once is strictly better
than answering the same question 494 times, and it leaves an auditable record of what
was excluded and why — which a fuzzy instruction to a model does not.

**Full category list, dispositions and the reasoning: [`taxonomy.md`](taxonomy.md).**
That document is the decision record; this one is the labelling guide.

The canonical positive is **makolet**. If a rule would exclude it, the rule is wrong.

## Settled cases

| Case | Decision | Reason |
| --- | --- | --- |
| Hebrew word inside an English sentence | **positive** | The core case |
| Multi-token institutional terms — *Bituach Leumi*, *teudat zehut*, *osek patur* | **positive**, one span | They are one unit and the TTS needs them as one |
| Hebrew term with English plural — *latkes*, *chagim* used as a plural | **positive**, span covers the whole surface form | The pronunciation problem is in the stem |
| Guttural ח / כ words | **positive** | English orthography cannot represent them |
| Anglicised Hebrew — *kosher*, *rabbi*, *Shabbat*, *hummus*, *kashrut* | **negative** | Already pronounced acceptably. Enforced by `data/anglicised.csv`, not by judgement |
| Hebrew function words and particles — *od*, *po*, *ma*, *mi*, *lo*, *ken* | **negative** | Confirmed by Daniel 2026-09-22: there is no context in which these appear untranslated in English speech. They entered the term list as substring artefacts |
| A lexicon term matching inside a longer word — *ma* in *moshav* | **negative** | This is the source dataset's defect; do not reproduce it |
| English words that happen to be quarantined terms — *at*, *hi*, *lo*, *baby* | **negative** when used as English | Quarantine exists to catch exactly these |

## UNDECIDED — settle these early

**Place names.** *Tel Aviv*, *Jerusalem*, *Haifa*, *Eilat* are pronounced acceptably
in English and are probably negatives. But *Ashdod*, *Ashkelon*, *Palmachim*,
*Hof HaBonim*, *Galil* are less certain, and the source lexicon contains all of them.
A consistent rule is needed, not case-by-case instinct.

*Provisional:* negative if it has a conventional English pronunciation, positive
otherwise. Record what you actually do.

**Street and road names.** *Dizengoff*, *Ben Yehuda*, *Ayalon*, *Rothschild*,
*Hahagana*. Note *Rothschild* is Germanic, not Hebrew, and is already quarantined as
an English homograph.

### Brand and company names — DECIDED 2026-09-22: positive by default

**Israeli brand and company names are positives.** Decided by Daniel: *Yad2* is
pronounced as Hebrew, so it should be read as Hebrew, and including brand names is the
safer option — a brand read in Hebrew by an Israeli-context podcast is right far more
often than it is wrong.

The dataset's 42 unlabelled records are full of them: *10 Bis*, *Zol Stock*,
*Max Stock*, *Super Pharm*, *Rami Levy*, *Aroma*, *Cellcom*, *Yad2*, *Ynet*, *Mako*,
*Arutz 12*, *Galgalatz*, *Kan Bet*, *AM:PM*, *Tubi60*, *El Al*, *Janglo*, *Castro*,
*Fox Home*, *Nefesh b'Nefesh*.

Apply the default, and override only where the operational test clearly says otherwise
— a brand whose name is ordinary English words pronounced in English (*Fox Home*,
*Max Stock*) is a negative.

| Brand shape | Decision | Example |
| --- | --- | --- |
| Hebrew word or coinage | **positive** | *Galgalatz*, *Mako*, *Ynet*, *Janglo* |
| Hebrew + numeral | **positive** — but see the numeral trap | *Yad2*, *Arutz 12*, *10 Bis*, *Tubi60* |
| Hebrew + English word | **positive** | *Kan Bet*, *Zol Stock*, *Super Pharm* |
| Ordinary English words, said in English | **negative** | *Fox Home*, *Max Stock*, *AM:PM* |
| International brand, said in English | **negative** | *Castro* (as pronounced), *Aroma* |

**The numeral trap — unresolved, and it belongs to pass 3, not here.** *Yad2* is said
*yad shtayim*; *Arutz 12* is *arutz shteim-esre*. A digit handed to a Hebrew TTS
segment will be read as a Hebrew numeral, which is usually right — but *10 Bis* is
said *ten bis*, with the number in English. Annotate the span as Hebrew regardless;
record the spoken form in the lexicon's notes so pass 2 can emit the right Hebrew, and
let pass 3 keep digits out of segments where they would be read in the wrong language.
See [`pipeline.md`](pipeline.md).

**Proper names of people.** *Netanyahu*, *Herzog*. Probably negative for well-known
figures. Not yet represented in the corpus.

## Span boundaries

- Spans cover the **surface form** as it appears, including English affixes
  (*latkes*, *chagim*, *supermarkets* if it were positive).
- Spans do not include the preceding article, even when Hebrew takes one: tag
  *makolet*, not *the makolet*, and not *ha-makolet* unless *ha-* is actually written.
- Punctuation is outside the span.
- Adjacent Hebrew words that form a unit are one span (*Bituach Leumi*); adjacent
  Hebrew words that do not are separate spans.

## Reviewing against the machine pass

`scripts/annotate.py` proposes; it never decides. Two rules keep that honest:

1. **Only positively accepted spans become `B-HE`/`I-HE`.** Anything unreviewed or
   rejected stays `O`. An incomplete review yields an under-labelled corpus, which is
   a known quantity, rather than a wrong one.
2. **Typed terms that cannot be word-boundary matched are reported, not dropped** —
   `data/gold/report.json`. A typo in the review UI surfaces instead of vanishing.

The 360 spans that come from an aligned dataset label are pre-accepted and not
queued. That is a deliberate trade: it assumes an aligned label is correct, which is
a weaker assumption than trusting the labels wholesale, but not a free one. If the
review turns up aligned labels that are also wrong, stop and re-queue them.

## Label scheme v2: DECIDED 2026-11-15

Decided by Daniel on 2026-11-15, in a session about the span-review project in Carrot Cake Lab
(Label Studio project 35). **Not implemented yet:** the corpus, `adjudicate.py`, `build_splits.py` and
the released model are all still on v1 (a single tag, `HE`). The work list is in the
*Implementation order* table below.

**Principle: the dataset is richer than the model.** Every distinction below is annotated in the corpus.
Which of them a given model learns is decided at training time, so one dataset can serve several
model variants. This is experimental work, so a first model that is shaky on the new classes is
acceptable.

### Span tags (per span, in the corpus)

| Tag | Meaning | Examples |
| --- | --- | --- |
| `HEB` | Romanized Hebrew that is not established in English. This is v1's `HE` | *makolet*, *mashkanta*, *misrad hapnim* |
| `SHARED` | A Hebrew (or Yiddish) word also established in English. These were v1's "always negative" `anglicised` class | *Shabbat*, *kosher*, *kibbutz*, *chutzpah*, *schlep* |
| *(none)* | Everything else, **place names included** | *Tel Aviv*, *Ramat Gan*, *Dizengoff* |

**Place names stay untagged.** Daniel's reasoning: a place-name class drifts away from the classifier's
purpose. The scope stays narrow: romanized Hebrew, plus the shared/dual class. The other conservative
exclusions in [`taxonomy.md`](taxonomy.md) (`function_word`, `israel_english`, `asr_handles_it`) are
unchanged.

### Span attributes (per term, in the lexicon; copied onto each corpus span)

These are properties of the word, not of the sentence. They are therefore set once per term in
`data/terms.csv`, not tagged in every sentence. **There are two independent attributes, not one
combined class** (Daniel, 2026-11-15): the Hebrew variant matters more and must not be bundled with
language identification.

| Attribute | Values | Default | Notes |
| --- | --- | --- | --- |
| `lang` | `hebrew`, `yiddish`, `aramaic`, `arabic` | `hebrew` | The language the speaker is switching into, **not the word's origin**. Arabic-origin Israeli slang (*yalla*, *sababa*) is `hebrew` (DECIDED 2026-11-15). |
| `hebrew_variant` | `standard`, `ashkenazic`, `sephardic` | `standard` | Only when `lang` = `hebrew`; empty otherwise. This is the transliteration tradition the spelling signals: *Shabbos* / *Shabbat*, *-os* / *-ot* endings, *Succos* / *Sukkot*. A spelling that gives no sign is `standard`. |

**Model card note (required when anything trained on `lang` is published):** classing Arabic-origin
Israeli slang as `hebrew` is a simplification for this purpose, not a political or linguistic
judgement. It records what the speaker is switching into, not where the word came from.

**Consequence for spelling variants:** variants that signal a different `hebrew_variant` (*Shabbos* vs
*Shabbat*) become separate lexicon entries. They are no longer merged under one canonical spelling;
see *Spelling variation* in [`taxonomy.md`](taxonomy.md). Variants that are only orthographic
(*chag* / *hag*) stay merged.

### What a model may use

| Variant | Learns | Notes |
| --- | --- | --- |
| v1 (released) | `HE` = `HEB` | Unchanged; `SHARED` maps to O |
| v2 candidate | `HEB`, `SHARED` as two BIO types | Head grows by about 768 weights per extra label (negligible); the real cost is examples per class |
| attribute stage | `lang` and `hebrew_variant`, each its own output, from lexicon lookup; a spelling-cue classifier for unknown terms only if lookup proves insufficient | Not decided |

### Implementation order (tracked in Annotation-Environment `planning/backlog.md`)

| # | Step | Cost |
| --- | --- | --- |
| 1 | Relabel existing anglicised spans as `SHARED`: the 126 `negative:anglicised` sentences plus every occurrence of an `anglicised` term inside other sentences (lexicon lookup), then human check | none |
| 2 | Run the released model over the whole corpus (4,737 sentences); queue model-vs-label disagreements | cents (Modal / Replicate, not the laptop) |
| 3 | Rebuild LS project 35 as **one task per sentence**: all current spans pre-drawn, the reviewer adds missing spans and deletes wrong ones, with `HEB` and `SHARED` as labels. Queue: the 183 open span tasks, then step 2's disagreements, then the whole 468-sentence test split (for a trustworthy F1) | none |
| 4 | LLM-fill `lang` and `hebrew_variant` for every term (~800 terms); human review of uncertain picks, one decision per term | under $1, estimate first |
| 5 | Generate ~150-200 more `SHARED` terms × 3 sentences (~500-600 sentences); the current 68 terms are too thin | a few dollars; needs Daniel's approval of model and estimate |
| 6 | Train a v2 candidate on HF Jobs; v1 stays the published default until v2's numbers are read | cents |
