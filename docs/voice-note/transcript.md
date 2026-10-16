# Voice note: why this model exists

Audio: [`why-this-model.mp3`](why-this-model.mp3), 6 min 30 s, recorded by the author
on 2026-10-16, the day after release. Also on the
[Hugging Face model card](https://huggingface.co/danielrosehill/hebrew-latin-token-classifier).

Transcribed with AssemblyAI (`universal`, English) on 2026-10-16. The text below is
lightly edited: the corrections are listed at the end. The unedited ASR output is in
[`transcript-raw.txt`](transcript-raw.txt).

---

So the purpose of this voice note is to give a little bit of context about this
particular project, and the need for it as I saw it. The need was quite simple: I
couldn't find that it already existed, and thought it would be helpful for a project
which I'll explain. This is a classifier for identifying romanized Hebrew. So, Hebrew
words that are written in the Latin alphabet, the English-language alphabet, that
appear in text. I'll explain what might seem like a very obscure use case or a very
niche model, but it wasn't something that I could find, and there have been some
really useful small models developed for the other direction.

To do that, I'll tell the story of the project that I built this for. For about the
past year and a half, I've been developing and producing a podcast called My Weird
Prompts. The basic idea was that I sent in prompts, often specific prompts about AI
engineering, and never got around to reading the lengthy responses, and thought I'd
love to have that in audio format. So this podcast is a pipeline, a Python pipeline
running in Modal, and I send in my questions as text.

We generate using Chatterbox TTS, which is an English model, or at least we use it
with English. Because many of my prompts touched upon subjects related to living in
Israel, the generated text from the script-writing agent would just contain Hebrew
words written in English. You can predict what goes wrong. If the Hebrew words, like
*Shabbat* or *kosher*, are sufficiently well populated in the training set of an
English-language model that the phonetics are known, at least the phonetics as
they're pronounced by English speakers, they'll be pronounced in at least an
intelligible way. But there is a huge number of words that are just slightly too far
down the long tail, and which are kind of essential to having conversations about
Israel. We might talk about the National Insurance body, or about *Misrad
HaKalkala*.

Now, of course we could have a heuristic: always use a translation. But sometimes
there isn't really a good English translation. So I was thinking about how I could
actually go about improving on this. There are different approaches, like language
detection, of course, but it's a very tricky case when you have a script-generation
agent emitting text in the Latin alphabet; that's what it does. Because it does
that, the fact that the text is written in that alphabet is strongly suggestive, even
to these models, that it's English text, especially from the surrounding text, and
it's not expecting code-switching, which is, I guess, what's happening in this case.

Essentially, the written script itself contains code-switching. So I was thinking
about different ways this could be tackled, and I started breaking it down: what's
the problem here? The first problem I could think of was that it's seeing the script
and assuming everything is English. So, a first step: let's flag what's actually
Hebrew that's been written in English characters.

That's what this model is intended to do. And I'll explain the intended use case,
because it doesn't stop here. This pass would flag all the Hebrew words. The next
pass, as I envision it in the pipeline (and thankfully there's an existing model for
this), takes the word that's been flagged, let's take *[word]* for argument's sake,
and renders it in Hebrew characters.

Then the final step would be attaching the language tag to the span, saying it's
Hebrew. So now a model that's expecting Hebrew receives text that is actually in
Hebrew. That's the intended pipeline: those two accessory models, and in my case
using concatenation. For code-switching it would actually involve concatenation
within sentence boundaries, which is quite challenging, of course, but I think it
would be useful.

So that's the purpose of this model. In terms of validation, from some initial runs it
mostly works: a good initial benchmark. The failure modes are subtle to think about,
because you get to the question of *Shabbat*. How do you handle *Shabbat*?

There are two approaches to it, I guess. One approach says: *Shabbat* is a Hebrew
word, so it should be classified as Hebrew. But the implication is that, in this use
case at least, it's then going to be rendered by a Hebrew TTS engine. The other way to
look at it is that *Shabbat* has entered the English lexicon and shouldn't be
classified. Or maybe the classifier, in a future iteration, has a third type: Hebrew
words that are predictably in the English corpus, so that a TTS model just running in
English mode is not going to have trouble with them, and it's left as a user setting
how you want those words to be handled. But for the proof of concept that this
represents, I think it's probably a moot point, because hopefully it should work
reliably either way. The intention was to catch words like "I need to go to the
*Misrad HaBriut* today", "I went to the …", etc.

All these words: to have that word-level classification built into the pipeline and
get proper phonetic rendering in code-switching examples. That's the purpose of the
Latin classifier. Comments, feedback and ideas for the pipeline are more than welcome,
publicly at danielrosehill.com.

---

## Edits made to the ASR output

| ASR output | Edited to | Basis |
| --- | --- | --- |
| `chatterboxtts` | Chatterbox TTS | Product name |
| `Shabbat kosher` | *Shabbat* or *kosher* | Two example words, not a phrase |
| `the Misraad Ta Kalkala` | *Misrad HaKalkala* | Ministry of Economy; ASR split the prefix. Confirmed from context |
| `the misraad ibrute` | *Misrad HaBriut* | Ministry of Health; ASR split the prefix. Confirmed by the author 2026-10-16 |
| `Let's take, for argument's sake, and render that` | `let's take [word] for argument's sake, and renders it` | A word is missing from the ASR output; the example word was not recoverable |
| `the next flag` | the next pass | Context: the three-pass pipeline. The spoken word was "flag" |
| `danielrosehill. Com` | danielrosehill.com | Domain |

Everything else is filler removal ("um", repeated words) and punctuation. Nothing was
added to the substance.

The note treats *Shabbat* as an open question. The released model's annotation policy
already settles it the second way (anglicised words such as
*Shabbat*, *kosher*, *hummus* are **not** flagged; see
[`../annotation-policy.md`](../annotation-policy.md) and
[`../taxonomy.md`](../taxonomy.md)). The third-class idea is a possible future
direction, not current behaviour.
