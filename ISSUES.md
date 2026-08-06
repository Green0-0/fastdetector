# Dataset Issues — `cc-2020-rewritten` / `cc-2021-rewritten`

Consolidated from [audit.md](audit.md) and [audit2.md](audit2.md), **re-verified against the published
parquet files**. Every number below was recomputed; audit claims that did not hold are listed in
[§ Claims that did not survive verification](#claims-that-did-not-survive-verification).

**Verification scope**

| Dataset | Rows checked | Coverage |
| --- | --- | --- |
| `G-reen/cc-2021-rewritten` | 23,716 | all 7 shards, 5 generator models |
| `G-reen/cc-2020-rewritten` | 274,312 | shard_0 (granite, 137,033) + shard_2 (Qwen3-8B, 137,279) of 7 |

Row references are `shard_N[i]`, where `i` is the 0-based index within that shard.

---

## Severity summary

| # | Issue | cc-2020 | cc-2021 | Kind |
| --- | --- | --- | --- | --- |
| 1 | Whitespace normalization asymmetry (perfect label leak) | 26.0% | 9.8% | **P0 — leak** |
| 2 | Post-processing injects stray whitespace / drops content | 1.01% | 2.18% | **P0 — leak + loss** |
| 3 | `partial_encode` degenerate descriptors → pangram outputs | 312 rows | — | **P0 — garbage** |
| 4 | indirect_reference refusals / "you didn't give me the text" | 454 (0.17%) | 28 (0.12%) | **P0 — garbage** |
| 5 | Near-duplicate & verbatim pairs (rewrite, Qwen) | 17–28% | 10.8% | **P0 — no signal** |
| 6 | AI length scales sublinearly with input length | ratio 1.11→0.15 | same | INFO — generator property |
| 7 | Empty `final_response`, no retry/fallback | 93 | 5 | P1 — garbage |
| 8 | Model-specific formatting leakage (markdown / emoji / titles) | 18.7x bold | 6–79% by model | P1 — leak |
| 9 | Task meta-commentary in AI text | 10x / 19x | 5x / 13x | P1 — leak |
| 10 | Prompt instruction echoed into the AI text | — | 62 (0.26%) | P1 — garbage |
| 11 | Unfilled template placeholders | 1,452 (0.53%) | 257 (1.08%) | P2 — garbage |
| 12 | `use_multiturn=False` on 100% of rows (49% are 2-turn) | 274,312 | 23,716 | P2 — root cause of #3/#4 |
| 13 | Duplicate originals + train/test corpus overlap | 1,428 (0.52%) | 98 + 44 shared with train | P2 — contamination |
| 14 | Encoding residue (U+FFFD, literal `\n`, mojibake) | 399 / 287 | 26 / 23 | P2 — noise |
| 15 | Cross-shard model & param heterogeneity (documentation only) | 2 models | 5 models | P2 — docs |

---

## P0 issues

### 1. The human side is whitespace-normalized and the AI side is not — a 100%-precision label leak

This is the single most damaging problem in both datasets, and neither audit found it.

Measured on cc-2020 shard_0 (137,033 rows):

| Feature | in `final_response` (AI) | in `original` (human) |
| --- | --- | --- |
| `\n\n` (blank line) | common | **0 rows** |
| trailing space before a newline (`[ \t]\n`) | 34,430 (25.1%) | **0 rows** |
| two spaces inside a line (`\S  +\S`) | 117 | **0 rows** |
| leading whitespace | 2,619 (0.95%) | **0 rows** |
| trailing whitespace | 2,423 (0.88%) | **0 rows** |

A single regex — `[ \t]\n` OR edge-whitespace — flags **26.0% of AI rows and 0.00% of human rows**.
That is a free 26%-recall / 100%-precision detector that has nothing to do with writing style. On
cc-2020 shard_2 (Qwen) the `[ \t]\n` rate alone is **37.5%**. On cc-2021: `  \n` appears in 2,335
finals vs **0** originals; `\n\n\n` in 745 vs **0**.

**Example** — cc-2020 `shard_0[1]`, `final_response`:
`"...shortcuts are invaluable. \n\nA smarter estimate could reco..."` — note the space before `\n\n`.
No human row in the corpus can contain that byte sequence.

**Root cause.** The human text is not raw web text. [filter.toml](config/filter.toml) reads
`trafilatura_text`, and [gen/shard_0.toml](config/gen/shard_0.toml) then reads `collected_subset` —
which is produced by `is_loose_subset` at
[statistics_basic.py:293](src/fastdetector/statistics/statistics_basic.py#L293) as
`a_str[orig_start:orig_end+1]`, where the offset map is built only over **non-whitespace** characters
([statistics_basic.py:273-279](src/fastdetector/statistics/statistics_basic.py#L273-L279)). So the
slice is guaranteed to begin and end on a non-space character, on top of trafilatura's own whitespace
normalization. The AI side is raw model output, never normalized.

**Fix direction.** Apply the identical normalizer to both columns before training (or before upload),
not just to the human side.

---

### 2. `post_process_response` injects stray whitespace and silently drops real content

517 cc-2021 rows (2.18%) and 2,776 cc-2020 rows (1.01%) have a `final_response` matching neither
`response_0` nor `response_1`. All 517 are substrings of the last response — confirming this is the
`---`-stripping pass in [gen.py:9-83](scripts/gen.py#L9-L83) and not a selection step. Two defects:

**2a. Stray whitespace.** 451 of the 517 (87%) now *begin* with a newline and 434 end with
whitespace, because [gen.py:28](scripts/gen.py#L28) (`resp.split("---")[1]`) and
[gen.py:34](scripts/gen.py#L34) (`resp[:resp.rfind("---")]`) never re-strip. Since **zero** originals
have edge whitespace (issue #1), post-processing manufactures a perfect AI marker on the very rows it
was meant to clean.

**Example** — cc-2021 `shard_0[186]` (direct_reference, granite):
- `response_0`: `"**Course Overview: Mastering Digital Marketing for Small Businesses**\n\n---\n\n**What's Included?**\n\n- **180 Learning Minutes**..."`
- `final_response`: `"\n\n**What's Included?**\n\n- **180 Learning Minutes**..."`

**2b. Content loss.** Rule 1 keeps only the text *between* the first two `---`, so anything after the
second `---` is discarded. 165 rows lost more than 20 characters of genuine trailing prose. The
rule-5 revert guard ([gen.py:72](scripts/gen.py#L72), `>40 words or >25%`) is too coarse to catch
short-but-real tails.

**Example** — cc-2021 `shard_6[2307]` (revise) lost a 262-char closing paragraph:
`"\n\n---\n\nThis explanation aims to make the concept of fast debt settlement accessible and relatable, highlighting its importance..."`

---

### 3. `partial_encode` produces empty descriptors, and the second turn answers with a pangram

**306 of 137,033** cc-2020 shard_0 rows have `final_response` containing "The quick brown fox jumps
over the lazy dog"; **202 contain nothing else**. 305 of 306 come from the `partial_encode`
subcategory of `indirect_reference`. Across shards 0+2: 312 rows.

**Mechanism, traced end to end** (cc-2020 `shard_0[59]`):
1. Turn 0 instruction ([partial_encode.json](sample_prompts/indirect_reference/partial_encode.json)):
   *"…replace all nouns and verbs with their part-of-speech tags (e.g., [NOUN], [VERB]), keeping only
   adjectives, conjunctions, prepositions, articles, and punctuation intact."*
2. `response_0` = `"[NOUN] [VERB] [ADJECTIVE] [CONJUNCTION] [PREPOSITION] [ARTICLE] [PUNCTUATION]"` —
   a legend, containing zero information about the document.
3. Turn 1 asks to *"recreate the original human written text it describes as accurately as possible."*
   Because `use_multiturn=False` (issue #12), the model **never sees the source document** — only that
   legend.
4. `final_response` = `"The quick brown fox jumps over the lazy dog."`, paired with an original that
   begins *"So we're back – with another interview for our Youth Week Writerly Types series!"*

Same failure at `shard_0[1025]`, `shard_0[1466]`, `shard_0[1978]`, `shard_0[2072]`, `shard_0[2400]`.

---

### 4. `indirect_reference` refusals: the model complains the source text is missing

cc-2020 shards 0+2: **454 rows (0.17%)** open with a refusal, **395 of them indirect_reference**.
cc-2021: **28 rows (0.12%)**, all indirect_reference. Same root cause as #3 — turn 1 receives only
`response_0`, and when that descriptor references "the original passage", the model asks for it.

**Examples**
- cc-2021 `shard_0[31]` (granite): *"I'm unable to proceed with the task as requested because you
  haven't provided the original passage about creating and managing projects in a survey mapping tool.
  Please provide t…"*
- cc-2021 `shard_0[1162]`: *"**[Original Passage Placeholder]**\n\n*Since the original passage is not
  provided, I am unable to generate a response that adheres to the specified constraints.*"*
- cc-2020 `shard_0`: *"I'm sorry, but there doesn't appear to be any actual content or descriptor
  provided in your message."*

These rows teach a classifier to detect the phrase "I cannot" rather than AI prose.

---

### 5. Near-duplicate and verbatim pairs, concentrated in `rewrite` and in Qwen3-8B

| Measure | cc-2020 shard_0 (granite) | cc-2020 shard_2 (Qwen3-8B) | cc-2021 (all) |
| --- | --- | --- | --- |
| `final_response` == `original` exactly | 238 (0.17%) | **2,285 (1.66%)** | 77 (0.32%) |
| Jaccard(orig, final) > 0.9 | — | **28.2%** | 10.8% |

In cc-2021, 398 of the 538 high-Jaccard rows are `rewrite`. Per-prompt-stem exact-copy rates on
cc-2020 shard_0:

| rewrite instruction stem | exact copies |
| --- | --- |
| "Edit this text by replacing a singular section, without changing anything else." | 48/1,712 (2.8%) |
| "…the new text must contain at least half of the original text, unmodified, repeated verbatim." | 28/1,715 (1.6%) |
| "Edit this text by inserting exactly one to three sentences…" | 0/1,715 (0.0%) |

A pair where the two sides are identical carries no label information at all; a pair at Jaccard 0.9+
carries almost none. The `rewrite` prompt family asks for minimal edits by construction, and
Qwen3-8B interprets "minimal" as "none" an order of magnitude more often than granite does.

**Example** — cc-2021 `shard_2[3165]` (rewrite, Qwen), original 3,559 chars / final 3,278 chars,
both opening *"We were able to get two 25 pound boxes of peaches through the efforts of a friend…"*

---

## P1 issues

### 6. AI text scales sublinearly with input length (characterization, not a defect)

Median `len(final)/len(original)` on cc-2020 shard_0:

| original length | n | median length ratio |
| --- | --- | --- |
| < 2,000 chars | 57,827 | 1.11 |
| 2,000–5,000 | 57,615 | 0.90 |
| 5,000–10,000 | 15,953 | 0.54 |
| 10,000–20,000 | 4,563 | 0.30 |
| > 20,000 | 1,075 | **0.15** |

**This is signal, not error.** Instruct-tuned 8B models stopping after a few hundred to a couple of
thousand tokens is a genuine property of the generators. An earlier revision listed it as a P1 leak;
that was wrong. What the two columns' length distributions differ by at the tail is a fact about the
corpus, recorded here, not a defect to repair.

*It is not a token budget problem.* `max_tokens` is not passed
([generator.py:35-41](src/fastdetector/generator.py#L35-L41)), so vLLM already grants each request
`max_model_len` minus its prompt. Estimated over all 137,033 rows of cc-2020 shard_0 at 4 chars/token,
median headroom use runs 1.3% → 3.5% across the length buckets above. **Zero** rows use more than 80%
of their headroom, zero come within 500 tokens of the 32,000 limit, and the largest prompt-plus-output
observed is ~17,321 tokens. Neither setting `max_tokens` nor raising `max_model_len` would change
anything.

How far output scales with input is a **per-model property**, worth knowing when weighting the model
mix. Median output tokens on cc-2021, shortest bucket to longest:

| model | <2k | 2–5k | 5–10k | 10k+ | growth |
| --- | --- | --- | --- | --- | --- |
| gemma-4-E4B-it | 345 | 677 | 1,381 | 2,841 | **8.2x** |
| Qwen3-8B-AWQ | 381 | 694 | 1,408 | 2,819 | 7.4x |
| Ministral-3-8B | 412 | 676 | 1,030 | 1,441 | 3.5x |
| Llama-3.1-8B | 380 | 663 | 1,094 | 1,178 | 3.1x |
| granite-4.1-8b | 361 | 614 | 834 | 932 | **2.6x** |

Prompt wording moves it too: on >10k-char originals, `rewrite` ("as long as possible") reaches a
median 2,646 output tokens against `indirect_reference`'s 983.

The one thing still worth filtering is a *failed* generation as opposed to a short one — a 5,000-char
source answered with 200 characters. `is_empty` already covers the bulk of those; `has_length_anomaly`
exists for the rest and is deliberately unused by `gen.py`.

The `P(human | text > 42,000 chars) = 98.1%` figure quoted in earlier revisions came from the human
corpus admitting documents longer than any generator produces. It is a property of the source corpus'
length range, not of AI text.

### 7. Empty `final_response` when the last turn fails, with no retry or fallback

cc-2020: 93 empty finals (14 in shard_0, 79 in shard_2). cc-2021: 5. Plus 8 empty `response_0`, all
in the Qwen shard.

`final_response` is not a selected "best" response — it is unconditionally the **last** turn's output
([generator.py:266](src/fastdetector/generator.py#L266)), and a failed request returns `""`
([generator.py:53](src/fastdetector/generator.py#L53)). So a two-turn row with a perfectly good
`response_0` is published with an empty AI side whenever turn 1 fails.

**Examples**
- cc-2021 `shard_0[2455]`: `response_0` is a full generated prompt, `response_1` = `""`, final = `""`.
- cc-2021 `shard_2[1963]`: `response_0` = `"A ...  \nA ...  \nA ...  \nA ..."` (degenerate), final = `""`.
- cc-2021 `shard_2[1757]`: both responses empty — complete generation failure.

### 8. Model-specific formatting leakage

cc-2020 (shards 0+2, 274,312 rows):

| Feature | AI | human | ratio |
| --- | --- | --- | --- |
| markdown link `[t](u)` | 0.95% | 0.02% | **56x** |
| code fence ` ``` ` | 0.37% | 0.01% | **42x** |
| bold `**` | 22.02% | 1.18% | **19x** |
| `#` heading | 1.35% | 0.14% | 10x |
| numbered list | 7.99% | 4.15% | 1.9x |
| emoji | 2.18% | 1.13% | 1.9x |
| bullet list | 18.05% | 18.90% | 1.0x — clean |

This is *per-model style*, not "AI style". cc-2021 spread across its 5 generators:

| model | bold | AI-added title line | emoji injected | stray edge-ws |
| --- | --- | --- | --- | --- |
| Ministral-3-8B | **78.9%** | **48.5%** | 5.0% | 9.2% |
| granite-4.1-8b | 25.1% | 10.0% | 3.8% | 2.5% |
| gemma-4-E4B-it | 11.9% | 2.2% | 2.9% | 0.3% |
| Llama-3.1-8B | 9.4% | 6.3% | 0.0% | 0.2% |
| Qwen3-8B | 6.4% | 2.1% | 0.3% | 0.4% |

**2,882 cc-2021 rows (12.15%)** open with a bold or `#` title line the original does not have — 186
of which are *task* titles that name the pipeline:

- `shard_0[48]`: `"**Recreated Text:**\n\nThe inception of the current administration in Ogun State…"`
- `shard_0[51]`: `"**Rough Draft: Helping Friends Through Grief**\n\nHey there,…"`
- `shard_0[496]`: `"**Rough Draft Typed on a Phone**\n\n---\n\nTechNova Inc. announces…"`

Emoji injected into the AI text but absent from the original: **638 rows (2.69%)**, concentrated in
Ministral (270) and granite (261) and essentially absent from Llama (1) — e.g. `shard_0[0]`
`"Hey there! 🐾\n\nIt's been a wild June, huh?…"`.

### 9. Task meta-commentary bleeds into the AI text

| Pattern | cc-2020 AI | cc-2020 human | cc-2021 AI | cc-2021 human |
| --- | --- | --- | --- | --- |
| "In summary / To summarize / In conclusion" | 4.30% | 0.41% | 2.12% | 0.43% |
| "the original/above/provided text\|passage" | 0.96% | 0.05% | 1.06% | 0.08% |

**Example** — cc-2021 `shard_0[89]`: `"…Recreated Text:  \nMust wear a festive Christmas skirt this
upcoming year.  \n\nNote: The original text likely expressed a personal or stylistic decision to
incorporate a Christmas-th…"` — the model is narrating the rewriting task inside its own output.

### 10. The prompt instruction is echoed verbatim into the AI text

**62 cc-2021 rows (0.26%)** contain a ≥60-character verbatim slice of their own instruction; 46
contain one of the literal instruction tails, including 29 with `"Output the full new text with no
extra statements or commentations."`

**Example** — cc-2021 `shard_0[97]` (rewrite, granite), `final_response` ends:
`"…We'll do the rest and find the best offers for your search.\nModify this text to be as long as
possible. However, the new text must contain at least half of the original text, unmodified, repeated
verbatim.\nOutput the full new text with no extra statements…"`

Concentrated in `rewrite` (27) and granite (40). These rows are unusable as AI prose samples.

---

## P2 issues

### 11. Unfilled template placeholders

cc-2021: **257 rows (1.08%)** contain a bracketed placeholder absent from the original —
`[Your Name]` ×76, `[Company Name]` ×37, `[Name]` ×22, `[Date]` ×13, `[insert link]` ×7, and a long
tail. cc-2020 (0+2): 1,452 rows (0.53%).

15 cc-2021 rows are placeholder-only failures under 50 characters:

| Row | `final_response` | original length |
| --- | --- | --- |
| `shard_3[375]` | `"[Insert Source Text Here]"` | 3,693 |
| `shard_3[3127]` | `"[Source Text to be provided here]"` | 1,542 |
| `shard_3[3105]` | `"(Source Material Not Provided)"` | 15,215 |
| `shard_6[181]` | `"Down BEARISH"` | 695 |
| `shard_3[665]` | `"I cannot fulfill this request."` | 6,425 |

**Example with context** — cc-2021 `shard_0[344]` (revise) ends:
`"…the spirit of Dominican warmth and hospitality continues to thrive. ☀️🍹\n\nWarm regards,  \n[Your Name]"`

### 12. `use_multiturn` is `False` on every row, and that is the root cause of #3 and #4

Verified: **274,312/274,312** cc-2020 rows and **23,716/23,716** cc-2021 rows have
`use_multiturn = False`, while 139,381 and 11,616 respectively have 2 chat turns.

This is *not* a metadata bug — it is hardcoded at
[build_prompts.py:52](scripts/prompts/build_prompts.py#L52) and
[build_prompts.py:104](scripts/prompts/build_prompts.py#L104). But the behavioural consequence is
real: with `use_multiturn=False`,
[generator.py:183-184](src/fastdetector/generator.py#L183-L184) sends **only the last user message**,
so turn 1 sees `response_0` plus a follow-up instruction and *never the source document*. When
`response_0` is uninformative, the model has nothing to work from — producing the pangrams (#3), the
refusals (#4), and the "[Original Passage Placeholder]" outputs (#11).

Median Jaccard(original, final) per `indirect_reference` subcategory shows how much is lost:

| subcategory | n | median Jaccard | median length ratio |
| --- | --- | --- | --- |
| `translation_roundtrip` | 1,494 | 0.469 | 0.96 |
| `partial_encode` | 1,486 | 0.274 | 0.74 |
| `descriptive_encode` | 1,481 | 0.213 | 0.94 |
| `prompt_encode` | 1,462 | 0.201 | 0.83 |

Related: 27 of the 1,494 `translation_roundtrip` rows (1.8%) never made it back to English —
e.g. `shard_1[2009]` (Llama-3.1-8B) opening `"《罗宁：最后的武士》 is a recently released roguelike fighting
game on App Store, developed by Dreamotion team…"`.

### 13. Duplicate originals, within each corpus and across the train/test boundary

cc-2020 is the training corpus and cc-2021 the test corpus, so there is no split inside either one.
That makes **cross-corpus** overlap the contamination that matters, and it is real. Comparing all
23,716 cc-2021 rows against cc-2020 shards 0+2:

| | rows | share of test |
| --- | --- | --- |
| test originals byte-identical to a train original | 44 (15 distinct texts) | 0.186% |
| test originals sharing a train original's first 300 chars | 113 | 0.48% |

Both are **floors**: only 2 of cc-2020's 7 shards were checked, so the full figure is plausibly
around 3x higher. The most repeated overlapping text appears 20 times in the test corpus — it is
the cookie-consent notice again. The overlap exists because both corpora are Common Crawl snapshots
and the same boilerplate is served by the same sites year after year.

Within each corpus, duplicates remain a diversity problem rather than a leakage one: cc-2020 (0+2)
has 1,428 rows (0.52%) in duplicate-original groups — a cookie notice ×76, a Getty *"Your Easy-access
(EZA) account…"* notice ×57 and ×25 (spelling variant); cc-2021 has 98 rows across 26 distinct
duplicated texts, and 245 rows (1.03%) are cookie/privacy boilerplate.

Note that duplicates are largely invisible per shard. On cc-2021, deduplicating each of the 7 shards
independently finds 36 repeats; deduplicating the pooled corpus finds **73**. `is_duplicate` therefore
has to run on whole corpora, and the train/test check has to run across them.

### 14. Encoding residue

| | cc-2020 (0+2) | cc-2021 |
| --- | --- | --- |
| U+FFFD in `final` / `original` | 51 / 348 | 3 / 26 |
| literal `\n` escape in `final` / `original` | 116 / 171 | 23 / 24 |
| mojibake (`Ã©`, `â€`, `Â¦`…) | — | 93 (0.39%) |

Mostly inherited from the web source, so it lands on the *human* side more often than the AI side —
a small leak in the opposite direction. Example: `"…Made Different AdjustableÃ¯Â¼ÂInf Occasion Will
Roman Family Day Process Birthdays…"`.

### 15. Cross-shard heterogeneity is real and undocumented

`cc-2020-rewritten` is **not** single-model (as [audit.md](audit.md) assumed — shard_2 had not been
published yet). shard_0 is granite-4.1-8b at `temp=0.6, top_p=0.9, top_k=40`; shard_2 is Qwen3-8B at
`temp=0.7, top_p=0.8, top_k=20, enable_thinking=false`. Their failure profiles differ sharply
(exact copies 0.17% vs 1.66%; empty finals 14 vs 79; bold 22.0% vs 9.2%; `[ \t]\n` 25.1% vs 37.5%).

`cc-2021-rewritten` spans 5 models over 7 shards with **4 distinct generation-parameter sets**
(temperature 0.6 / 0.7 / 0.7 / 1.25). Any per-shard split therefore also splits on model *and* on
sampling config. This needs to be in the dataset card, and shard assignment must not correlate with
the train/test split.

---

## Claims that did not survive verification

Do not spend cleaning effort on these.

1. **"Truncation: 16.47% (cc-2021) / 9.6% (cc-2020) of AI responses lack terminal punctuation."**
   The human baseline is *higher*: cc-2020 AI 9.24% vs human **15.29%**; cc-2021 AI 14.76% vs human
   **16.03%**. Genuine mid-sentence truncation (last token is a function word) is 125 rows (0.046%)
   and 10 rows (0.04%). Ellipsis endings: AI 2.03% vs human 2.74%. Missing terminal punctuation is
   normal for scraped web text, not an AI artifact. The real length problem is issue #6.

2. **"Prompt leakage in `response_0`, 720 rows, HIGH severity."** The count is right (722), the
   diagnosis is not: **715 of them are the `prompt_encode` subcategory**, whose turn-0 instruction
   literally reads *"come up with a prompt that is most probable to be the one used to generate the
   text"* ([prompt_encode.json](sample_prompts/indirect_reference/prompt_encode.json)). An
   instruction-shaped `response_0` is the intended output. Only ~7 rows are genuine misfires.

3. **"Repetition / degenerate outputs, 44 rows."** True consecutive block repetition is **2 rows**
   (0.01%), both gemma: `shard_3[1255]` and `shard_3[1381]`. The 44 figure counted markdown table
   dividers and `---` rules.

4. **"XXX filler text, 76 rows, HIGH."** 36 matches, 16 of which also appear in the *original* —
   they are adult-site copy (*"Hot Asian XXX"*), not placeholders. Not a pipeline defect.

5. **"88 refusal-like responses."** 28 verified (0.12%). The "I will not" bucket (32 rows) contains
   **zero** refusals: 31 rows contain the phrase, none at the start, all ordinary prose.

6. **"Markdown heading prefix, 245 rows (1.03%)."** Undercounted by 12x — **2,882 rows (12.15%)**
   open with a title line the original lacks (issue #8).

7. **"`[Insert...]` placeholders, 110 rows."** Undercounted — **257 rows (1.08%)** (issue #11).

8. **"Mojibake, 305 rows (1.29%)."** 93 rows (0.39%) under a stricter pattern; the audit's pattern
   over-matched legitimate accented text.

9. **"Most extreme length ratio 7.8x."** The largest `final/original` ratio in cc-2021 is **73.9x**
   (`shard_5[327]`: 145 chars → 10,713 chars). The 7.8x figure described the single *longest*
   response, not the largest ratio.

10. **"`final_response` is selected from the best of `response_0` / `response_1`."** There is no
    selection or scoring step. It is unconditionally the last turn's response
    ([generator.py:266](src/fastdetector/generator.py#L266)); the ~50/50 `r0`/`r1` split is just the
    single-turn/two-turn split.

11. **"Single generator model" / "shard 2 unavailable"** ([audit.md](audit.md)) — both obsolete;
    see issue #15.

12. **"Bold markdown, 35% AI vs 0.7% human (50x)."** Real but overstated: 22.02% vs 1.18% (18.7x)
    across shards 0+2. Still the largest *formatting* leak — though issue #1 is larger.

13. **"`use_multiturn` flag is non-functional / incorrectly set."** It is set deliberately and it
    does something; the problem is what it does. See issue #12.

**Claims that reproduced exactly** (no re-check needed): 517 post-processed rows; 217 rows under 0.1x
length ratio; 372 rows over 5x; 5 empty `final_response`; 8 empty `response_0` (all Qwen shard);
15 short responses; 12,104 empty `response_1`; ~673 emoji-injection rows (measured 638); 238 exact
copies in cc-2020 shard_0; the "quick brown fox" estimate of ~274 (measured 306 in shard_0).
