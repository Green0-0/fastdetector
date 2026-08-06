# Prompt and configuration fixes

The defects in [ISSUES.md](ISSUES.md) that no row filter can repair. A filter can
only delete the affected rows; the fixes here stop them being generated.

**Status: §2 to §5 are applied.** [sample_prompts/](sample_prompts/) and
[build_prompts.py](scripts/prompts/build_prompts.py) carry the changes, and
`prompts/` has been rebuilt from them. §1 needed no pipeline change; §6 and §7
are recorded as non-defects. [config/](config/) is untouched.

Ordered by how much data each one recovers.

---

## 1. Gating turn 1: which descriptors are actually broken

**Issue 12**, and the root cause of issues 3, 4 and part of 11.

`use_multiturn=False` is hardcoded at
[build_prompts.py:52](scripts/prompts/build_prompts.py#L52) and
[build_prompts.py:104](scripts/prompts/build_prompts.py#L104), so
[generator.py:183-184](src/fastdetector/generator.py#L183-L184) sends **only the
last user message**. For the 49% of rows with two chat turns, turn 1 receives
`response_0` plus a follow-up instruction and nothing else.

That is correct and must stay. The point of `indirect_reference` is that turn 1
works from the descriptor alone, so the AI text is not a copy of the source.
Restoring the conversation would defeat the design.

### The test cannot look for document content

`response_0` legitimately contains no document content in three of the four
subcategories: `translation_roundtrip` returns another language,
`prompt_encode` returns an instruction, and `partial_encode` returns a skeleton
with the content deliberately removed. Measured on cc-2020 shard_0, the
descriptors that are *working as designed* include:

- 2,129 rows whose `response_0` is non-Latin script,
- 1,178 rows whose `response_0` is a bare emoji sequence — one variant asks for
  exactly that, and it compresses a 4,165-character document into 36 characters.

Anything that tests for content, length, or a length ratio kills these. A ratio
floor of 0.05 catches 203 of the 305 pangrams at a cost of **2,024** healthy
rows, most of them the compressive descriptors above. That instrument is wrong.

### Two tests work, and they catch different failures

**Test A — the response is built from the instruction's vocabulary.** The
degenerate descriptors are the model echoing the *example embedded in its own
instruction* instead of applying it:

| instruction | `response_0` |
| --- | --- |
| "…replace all nouns and verbs with their part-of-speech tags (e.g. `[NOUN]`, `[VERB]`)" | `[NOUN] [VERB] [ADJECTIVE] [CONJUNCTION] …` |
| "…replace 3/4 of it with a singular blank underscore" | `_` |

Nothing is copied verbatim — `[ADJECTIVE]` and `[CONJUNCTION]` never appear in
the instruction — so `has_prompt_echo`, which looks for a verbatim span, cannot
see this. The right measure is **containment of the response's vocabulary in the
instruction**. Median containment on cc-2020 shard_0:

| rows | containment in instruction | containment in document |
| --- | --- | --- |
| all `indirect_reference` | 0.048 | 0.466 |
| pangram outcomes | **0.429** | **0.000** |

A 9x separation. `has_instruction_vocabulary` implements it.

**Test B — the descriptor collides with another row's descriptor.** A descriptor
byte-identical to another document's descriptor cannot encode either. This needs
no notion of language, format or content, because it compares descriptors only
to *each other*, never to the source.

Scoping is what makes B safe. Globally, low-alphabet encodings collide by
design; scoped to rows sharing a turn-0 instruction, a collision means that
instruction produced the same output for two different documents.
`is_duplicate(response_0, groups=instruction)` implements it.

### Measured, over the 34,260 `indirect_reference` rows

| gate | fires | pangrams caught | emoji descriptors | non-Latin | median Jaccard(orig, final) |
| --- | --- | --- | --- | --- | --- |
| A: instruction vocabulary | 613 (1.79%) | **299/305** | 5/1,178 | 1/2,129 | 0.024 |
| B: scoped collision | 789 (2.30%) | 208/305 | **0**/1,178 | **0**/2,129 | 0.037 |
| A ∪ B | 1,137 (3.32%) | 299/305 | 5/1,178 | 1/2,129 | 0.046 |
| *baseline, all rows* | — | — | — | — | *0.296* |

Two things to read off this. First, the gated rows have a median
Jaccard(original, final) of 0.02–0.05 against a 0.296 baseline — when either
gate fires, turn 1 really did recover almost nothing, so both are predicting a
bad *outcome*, not merely an unusual descriptor. Second, they are largely
disjoint: 524 rows fire on B alone, 348 on A alone, 265 on both. B catches the
descriptors with no word characters at all (`_`, empty), which score zero
containment by definition; A catches the unique-but-uninvertible ones, which
cannot collide. Use both.

What they gate is concentrated in two instructions:

| gated by B | instruction |
| --- | --- |
| 406/1,719 (23.6%) | "Take this text, but replace 3/4 of it with a singular blank underscore…" |
| 277/1,711 (16.2%) | "Strip this text down to its bare syntax. Replace all nouns and verbs…" |
| 81/2,144 (3.8%) | "Translate the given text to Hindi." |
| 21/1,441 (1.5%) | "Translate the entirety of this text into a sequence of emojis…" |

Every `prompt_encode` variant and most `descriptive_encode` variants gate at
0.00%.

### Gate after the fact, in gen.py — not mid-pipeline

An earlier draft of this document proposed a per-turn predicate in
[`build_dataset`](src/fastdetector/generator.py#L190) so a row could be dropped
between turns. Measured, that is not worth doing. Applying the same filters to
`final_response` *after* generation, over the same 34,260 rows:

| | rows removed |
| --- | --- |
| turn-0 gate on `response_0` (A ∪ B) | 1,137 (3.32%) |
| post-hoc filters on `final_response` | **3,011 (8.79%)** |
| of the turn-0 gate, already caught post-hoc | 915 (**80.5%**) |
| removed *only* by the turn-0 gate | 222 (0.65%) |

Filtering afterwards catches nearly three times as much, because it observes the
outcome instead of predicting it. And the 222 rows the early gate uniquely
removes are not all bad — sampled, they include perfectly good AI text whose
descriptor merely tripped a proxy:

```
response_0     'Winterfest at Mendon Ponds Park will take place on Sund…'
final_response 'Winterfest at Mendon Ponds Park will take place on Sunday, January 17,
                from 11 a.m. to 4 p.m. The event promis…'
```

Gating early would have destroyed that row. The compute it saves is ~1,137
turn-1 requests out of 137,033 rows, under 1%. **Leave the pipeline alone and
filter in `gen.py` after generation**, where `response_0` is still a column and
both signals remain available.

**Proposed scheme.**

1. In `gen.py`, after generation, drop rows on `final_response` using
   `is_empty`, `has_refusal`, `has_filler_output`, `has_length_anomaly` and
   `has_placeholder`. This is where the yield is.
2. Optionally add `has_instruction_vocabulary(response_0, instruction)` and
   `is_duplicate(response_0, groups=instruction)` on top. They are 80% redundant
   with step 1 and carry some false positives, so treat them as flag columns
   rather than deletions unless you want the extra 0.65%.
3. Their real value is **diagnostic**, not as row filters: run them per turn-0
   instruction and they tell you *which prompts to fix* — which is exactly how
   the two broken instructions in §2 were identified.

One gap this exposed, now closed: the models complain about a missing input in
prose that never says "I cannot" or "I'm sorry" — *"It appears that the input
provided is a placeholder line indicating where a trace of a human-written
document should be."* `has_refusal` now catches these, requiring the complaint
opener to be about the input; that adds **232** rows on cc-2020 shard_0 with
**zero** matches against its 137,033 originals.

`use_multiturn` being `False` on 100% of rows is not itself a bug — it just
needs to be in the dataset card so users do not read it as "single-turn".

---

## 2. Two turn-0 instructions are not invertible — APPLIED

**Issue 3.** The gate above is a net, not a repair. Two instructions account for
almost everything it catches, and both ask for an encoding that cannot be
inverted — turn 1 is then told to "recreate the original human written text as
accurately as possible" from it, which is not a task that has an answer.

**"Strip this text down to its bare syntax"**
([partial_encode.json](sample_prompts/indirect_reference/partial_encode.json)) is
the sole source of all 305 pangrams — 17.8% of the 1,711 rows using it. Models
frequently return the legend rather than the tagged document:

```
[NOUN] [VERB] [ADJECTIVE] [CONJUNCTION] [PREPOSITION] [ARTICLE] [PUNCTUATION]
```

Even when they do tag the document properly, every content word is gone, so the
descriptor is close to uninvertible by construction.

**"Replace 3/4 of it with a singular blank underscore"** collapses to `_` or
`_ _ _` on 23.6% of its 1,719 rows — the highest degenerate rate of any
instruction.

**Proposed scheme.** Retire both, or bound the information loss so the result is
still invertible. Proposed replacements, verbatim (these are literal prompt
strings, so they carry no markdown emphasis and no em dashes; putting either in
a prompt teaches the model to emit it, and both are among the strongest AI
signals in the data at 18.7x and 3.1x the human rate):

POS variant, keeping the nouns is what makes it recoverable:

```
Replace every verb in this text with the tag [VERB], keeping all nouns,
adjectives, articles and punctuation exactly as they appear. Reproduce the
document in full, in its original order and paragraph structure. Do not output a
legend or a list of the tags you used.
```

Underscore variant, capped at a redaction a reader could reconstruct:

```
Take this text and replace one word in every four with a single underscore (use
one underscore for each removed word, never several in a row). Reproduce the
rest of the document in full, in its original order and paragraph structure.
```

Worth a look separately: "Translate the entirety of this text into a sequence of
emojis" gates at only 1.5%, but its median Jaccard(original, final) is **0.081**,
the lowest of any instruction — it rarely fails loudly, it just rarely works.

---

## 3. The `rewrite` family asks for edits so small the model returns the input — APPLIED

**Issue 5.** Exact `original == final_response` copies: 238 (0.17%) on cc-2020
shard_0, **2,285 (1.66%)** on shard_2. Jaccard > 0.9 reaches **28.2%** on
shard_2. The two worst stems:

| stem | exact copies |
| --- | --- |
| "Edit this text by replacing a singular section, without changing anything else." | 48/1,712 (2.8%) |
| "…the new text must contain at least half of the original text, unmodified, repeated verbatim." | 28/1,715 (1.6%) |
| "Edit this text by inserting exactly one to three sentences…" | 0/1,715 (0.0%) |

The pattern is clear: stems that specify a **positive, countable** action
("insert one to three sentences", "split the three longest sentences") never
degenerate. Stems that specify a **bounded** action ("without changing anything
else", "at least half unmodified") invite the model to change nothing at all,
because doing nothing satisfies the constraint as written.

**Proposed scheme.** Rewrite the bounded stems so the edit is mandatory and
countable, and turn the preservation constraint from a floor into a range.
Verbatim replacements, again free of markdown emphasis and em dashes:

Replacing `"Edit this text by replacing a singular section, without changing anything else."`:

```
Rewrite exactly one section of this text so that its wording is entirely new,
and leave every other section untouched. The rewritten section must share no
sentence with the original.
```

Replacing the "at least half unmodified" stem:

```
Modify this text to be as long as possible. At least half of the original text
must be repeated verbatim, and at most three quarters of it; the remainder must
be new prose.
```

This has to be fixed at the prompt, because deduplication is not available as a
fallback in `gen.py`. Over-similar pairs are removed later by the analysis stage
via its own filter conditions (`cosdist >= 0.03`, `softngram >= 0.06` in
[analysis.toml](config/analysis.toml)), which is where that decision belongs:
it needs the distance statistics, and it is a corpus-level judgement rather than
a generation failure. `gen.py` therefore drops only rows whose generation failed
outright, and never compares the two columns for similarity.

---

## 4. Instruction tails get copied into the output — APPLIED

**Issue 10.** 62 cc-2021 rows (0.26%) reproduce a line of their own prompt
verbatim; 29 contain `"Output the full new text with no extra statements or
commentations."` — the instruction the pipeline appends at
[build_prompts.py:50](scripts/prompts/build_prompts.py#L50).

The tail is appended as a bare line with no separator, so it reads as part of the
document the model was handed:

```
{{DOC}}
{{TEXT}}
Output the full new text with no extra statements or commentations.
```

**Proposed scheme.** Delimit the document so the instruction cannot be mistaken
for content. Change the `force_reformat` templates at
[build_prompts.py:49-50](scripts/prompts/build_prompts.py#L49-L50) and
[build_prompts.py:101](scripts/prompts/build_prompts.py#L101) to:

```
<document>
{{DOC}}
</document>

{{TEXT}}
Output the full new text with no extra statements or commentations.
```

This also reduces issue 9 (meta-commentary), which shares the same cause: the
model cannot tell where the task ends and the document begins, so it narrates the
boundary.

---

## 5. The "rough draft on a phone" framing produces a literal title — APPLIED

**Issue 8.** 186 cc-2021 rows open with a task label — `**Rough Draft:**`,
`**Rough Draft Typed on a Phone**`, `**Recreated Text:**` — because the prompt
names the artefact it wants and the model dutifully labels its output.

`strip_added_title` removes these cleanly, so this is the lowest-priority prompt
change. If the prompts are revised anyway, add to the `direct_reference` and
`indirect_reference` tails:

```
Begin directly with the text itself. Do not add a title, a heading, or a label
naming what you have written.
```

---

## 6. Not a defect after all: the length asymmetry

Two earlier drafts of this section were wrong, and it is retained only to record
why, so the same proposals do not get re-raised.

**Short AI output is signal, not error.** Instruct-tuned 8B models stopping after
a few hundred to a couple of thousand tokens is a genuine property of the
generators. A detector may legitimately learn it.

**No knob changes it.** `max_tokens` is already `max_model_len` minus the prompt,
because vLLM defaults it that way when a request omits it, so not setting it was
never the bug. Raising `max_model_len` does nothing either: over all 137,033 rows
of cc-2020 shard_0, median headroom use runs 1.3% to 3.5%, **zero** rows exceed
80% of their headroom, and the largest prompt-plus-output observed is ~17,321
tokens of the 32,000 available. Granting more of a budget that is 97% unused
costs KV cache per sequence and buys nothing.

Worth keeping only as characterization: how far output scales with input is a
per-model property (gemma 8.2x and Qwen 7.4x from the shortest bucket to the
longest, against granite's 2.6x), and prompt wording moves it as well, with
`rewrite` reaching a median 2,646 output tokens on >10k-char sources against
`indirect_reference`'s 983. Both are useful when weighting the model mix, neither
is a bug. `has_length_anomaly` remains available for genuinely *failed*
generations and stays unused by `gen.py`.

---

## 7. Not a defect: shard and model heterogeneity

An earlier draft flagged that shard membership correlates with generator model
and sampling parameters, so a per-shard train/test split would also split on
model. That does not apply. cc-2020 is the training corpus and cc-2021 the test
corpus, so neither is split internally and shard membership never reaches a
split boundary.

What the corpus-level arrangement does create is a **cross-corpus contamination**
check that issue 13 now covers: 44 cc-2021 rows (0.186%) carry an original
byte-identical to one in cc-2020, and 113 (0.48%) share a 300-character prefix,
both measured against only 2 of cc-2020's 7 shards and therefore floors. The
cause is that both corpora are Common Crawl snapshots carrying the same site
boilerplate.

That check has to run across the two corpora, and deduplication within each has
to run on the whole corpus rather than per shard: on cc-2021, per-shard
deduplication finds 36 repeats against 73 for the pooled corpus. Neither is a
prompt or config change; both are `is_duplicate` applied at the right scope.


---

## Where each change landed

`load_raw_samples_balanced_autosplit` splits each sample-prompt file 80/20, so a
given instruction variant lives wholly in the train prompt set or wholly in the
test one. cc-2020 was generated from `combined_dataset_train.json` and cc-2021
from `combined_dataset_test.json`, which is why the pangram-producing instruction
appears 1,711 times in cc-2020 shard_0 and **zero** times in cc-2021.

| repaired instruction | prompt split | corpus it was breaking |
| --- | --- | --- |
| POS-tag variant (§2) | train | cc-2020 (1,711 rows) |
| underscore FIM variant (§2) | train | cc-2020 (1,719 rows) |
| section no-op stem (§3) | train | cc-2020 (2,548 rows) |
| verbatim-floor stems (§3) | both | cc-2020 (2,604) and cc-2021 (1,341) |
| document delimiters (§4) | both | both, all four prompt types |
| no-title instruction (§5) | both | both, all four prompt types |

Regenerating a corpus therefore has to point `prompt_file` at the split that
corpus was built from.
