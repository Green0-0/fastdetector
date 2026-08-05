## Dataset Quality Analysis Report

G-reen/cc-2020-rewritten

This report provides a comprehensive quality analysis of the HuggingFace dataset G-reen/cc-2020-rewritten, which is designed as a training dataset for a human vs. AI text classifier. Each row contains a human-written original text and a paired AI-generated response, with metadata columns reporting generation parameters. The dataset is split across shards 0 through 6 (shard 2 was unavailable at the time of analysis).

The analysis covers schema integrity, missing/empty values, text quality, boilerplate/artifact detection, prompt diversity, label leakage potential, structural markers, encoding issues, truncation patterns, and per-prompt-type quality metrics. All quantitative findings are derived from a full pass of shard 0 (137,033 rows) plus sampled deep analysis of 2,000-5,000 rows.

|  Property | Value  |
| --- | --- |
|  Dataset | G-reen/cc-2020-rewritten  |
|  Shards analyzed | shard_0 (full), shard_1/3/4/5/6 (streaming unavailable)  |
|  Total rows (shard_0) | 137,033  |
|  Columns | 7 (original, prompt, response_0, response_1, final_response, generator_model, generation_params)  |
|  Prompt types | 4 (direct_reference, revise, indirect_reference, rewrite)  |
|  Generator model | cyankiwi/granite-4.1-8b-AWQ-INT4 (100%)  |
|  Analysis date | 2026-08-03  |

## 1. Schema and Structure

The dataset uses a well-structured schema with seven columns. The original column holds the human-written source text, while final_response holds the selected AI-generated text that serves as the positive (AI) class for classifier training. The intermediate responses (response_0 and response_1) represent the two candidate generations, with final_response being selected from one of them. The prompt column is a nested structure containing the chat turns sent to the model, the multiturn flag, examples (unused), and metadata including PROMPT_TYPE.

### 1.1 Column Summary

|  Column | Type | Description | Notes  |
| --- | --- | --- | --- |
|  original | string | Human-written source text | Min 178 chars; well-populated  |
|  prompt | struct | Chat turns + metadata | Contains chat_turns, use_multiturn, examples, metadata  |
|  response_0 | string | First AI generation | 253 empty (0.18%)  |
|  response_1 | string | Second AI generation | 67,406 empty (49.2%) - single-turn only  |
|  final_response | string | Selected AI text for training | 14 empty (0.01%)  |
|  generator_model | string | Model identifier | Single model across all rows  |
|  generation_params | string | JSON generation parameters | Single config across all rows  |

### 1.2 Key Structural Findings

OK: All four prompt types are evenly distributed at ~25% each (34,234-34,272 rows per type). This balanced distribution is excellent for training a classifier that should generalize across manipulation strategies.

OK: The single generator model and single generation config across all rows is consistent and indicates a controlled generation setup. No model mixing means classifier features will be more consistent.

WARNING: response_1 is empty for 49.19% of rows (67,406), corresponding exactly to the 49.19% of rows with single chat turn (1 turn). The remaining 50.81% (69,632) have 2 chat turns and a populated response_1. This is expected: multiturn prompts produce two responses. However, the use_multiturn flag is False for ALL rows, which is inconsistent with the 50.81% of rows that actually have 2 chat turns and a response_1. This flag appears to be non-functional or incorrectly set.

WARNING: final_response differs from both response_0 and response_1 in 1.33% of cases (1,817 rows). This means the final response was post-processed or modified after generation, which could indicate additional filtering or editing applied to the selected response.

## 2. Text Length Analysis

Text length is a critical feature for classifier training. If AI and human texts have systematically different length distributions, the classifier may learn a trivial length heuristic rather than meaningful linguistic patterns. The following analysis examines the character length distributions for both the original (human) and final_response (AI) texts.

### 2.1 Length Distribution Statistics

|  Metric | Original (Human) | Final Response (AI)  |
| --- | --- | --- |
|  Minimum | 178 chars | 0 chars  |
|  5th percentile | 855 chars | 550 chars  |
|  Median | 2,297 chars | 2,165 chars  |
|  Mean | 3,320 chars | 2,497 chars  |
|  95th percentile | 9,086 chars | 5,299 chars  |
|  Maximum | 57,376 chars | 41,846 chars  |

### 2.2 Key Length Findings

WARNING: AI texts are systematically shorter than human texts. The mean AI length (2,497 chars) is 75.2% of the mean human length (3,320 chars). The 95th percentile shows an even larger gap: AI texts max out at 5,299 chars vs. 9,086 for humans. This length asymmetry could cause the classifier to learn length as a cheap proxy for class, especially for texts over 5,000 chars where almost all examples would be human.

WARNING: 14 final_response entries are completely empty (0 chars). These provide no AI training signal and should be removed. 24 final_response entries are under 20 characters, which is also extremely short for training.

The length ratio (final_response / original) has a median of 0.955, meaning AI texts are typically 4.5% shorter than their human counterparts. However, the distribution has heavy tails: 8.2% of rows have a ratio below 0.3 (AI text is less than 30% of original length), and 2.0% have a ratio above 3.0 (AI text is more than 3x the original). These extreme ratios likely correspond to prompt types that fundamentally transform the text (e.g., translation to a more verbose style, or summarization).

### 2.3 Extreme Length Cases

10 original texts exceed 50,000 characters. These very long human texts may represent edge cases such as full articles, legal documents, or concatenated content. No AI final_response exceeds 50,000 characters, confirming that the model truncates or compresses very long inputs. This is a systematic behavior that could leak as a feature: texts above ~42K chars are always human.

## 3. Boilerplate and AI Artifact Analysis

One of the most critical quality concerns for a human/AI classifier training dataset is the presence of boilerplate or AI-specific artifacts in the generated text. These artifacts are linguistic patterns that never appear in human-written text but are common in AI outputs, such as preambles ("Here's the rewritten text"), disclaimers ("Note that this is an edited version"), and self-references ("As an AI"). If present, these artifacts give the classifier trivially easy features that don't generalize to real-world AI text that has been stripped of such patterns.

### 3.1 Preamble Detection

|  Pattern | In final_response | In original | Leakage Risk  |
| --- | --- | --- | --- |
|  Sure/Certainly preamble | 16 (0.08%) | 13 (0.07%) | Low  |
|  Here's/Here is preamble | 54 (0.27%) | 196 (0.98%) | Low  |
|  I'd be happy to | 4 (0.02%) | 3 (0.02%) | Low  |
|  Of course!/Absolutely! | 24 (0.12%) | 55 (0.28%) | Low  |
|  Let me/Below is | 5 (0.03%) | 21 (0.11%) | Low  |

OK: AI preamble rates are very low and sometimes even lower than in human text. The basic filtering pass mentioned by the dataset creator has been largely effective at removing obvious AI preambles like "Sure, here's the text:". The rates are low enough that they should not dominate classifier training.

### 3.2 Meta-Commentary and Self-Reference

|  Pattern | In final_response | In original | Leakage Risk  |
| --- | --- | --- | --- |
|  Summary meta (In summary/To summarize) | 275 (5.5%) | 25 (0.5%) | HIGH  |
|  Overall meta-commentary | 358 (7.2%) | 266 (5.3%) | Moderate  |
|  Note/Please note/Keep in mind | 321 (6.4%) | 379 (7.6%) | Low  |
|  Original text/meaning reference | 70 (1.4%) | 7 (0.1%) | HIGH  |
|  The above/following text reference | 91 (1.8%) | 23 (0.5%) | HIGH  |
|  Preserve original meaning | 4 (0.1%) | 3 (0.1%) | Low  |
|  AI self-reference (as an AI) | 51 (1.0%) | 35 (0.7%) | Low  |
|  I/we have edited/modified | 1 (0.0%) | 6 (0.1%) | Low  |
|  AI limitation statement | 8 (0.2%) | 11 (0.2%) | Low  |
|  AI training/knowledge ref | 5 (0.1%) | 12 (0.2%) | Low  |

WARNING: Summary meta-commentary ("In summary", "To summarize", "In conclusion") appears in 5.5% of AI texts but only 0.5% of human texts - a 11x ratio. This is a SIGNIFICANT label leakage risk. A classifier could trivially learn that texts containing "In summary" are AI-generated. In real-world deployment, human authors do use these phrases, so this feature would cause false positives.

WARNING: References to the original text ("original text", "original meaning", "the above text", "the following passage") appear in 1.4-1.8% of AI texts but barely in human text. These are artifacts of the rewriting process where the AI comments on its transformation. They are strong leakage signals.

Most AI self-references ("as an AI", "language model") and limitation statements are present at similar rates in both human and AI text, likely because some original texts are themselves about AI topics. These do not pose significant leakage risk.

### 3.3 Social/Conversational Patterns

|  Pattern | In final_response | In original | Leakage Risk  |
| --- | --- | --- | --- |
|  Hope this/that | 37 (0.7%) | 67 (1.3%) | Low  |
|  If you need/want | 173 (3.5%) | 539 (10.8%) | Low (reversed!)  |
|  Any further/more help | 146 (2.9%) | 273 (5.5%) | Low (reversed!)  |
|  Thanks/Thank you | 300 (6.0%) | 389 (7.8%) | Low  |
|  Feel free to | 49 (1.0%) | 72 (1.4%) | Low  |

OK: Social/conversational patterns are often MORE common in human text than AI text, which is actually a healthy sign. The human-written originals in this dataset (from CommonCrawl) include many blog posts, marketing copy, and customer-facing content that naturally uses phrases like "If you need any more help" and "Feel free to". The AI texts, being more direct rewrites, use these less frequently.

# 4. Placeholder and Default Text

Placeholder text in AI responses is a severe quality issue because it means the model failed to generate meaningful content, producing a generic fallback instead. These entries provide no useful training signal for the classifier and may actively harm it if the classifier learns that short generic text is AI-generated (which would not generalize to real AI text).

WARNING: The pangram "The quick brown fox jumps over the lazy dog" appears as the final_response in approximately 10 out of 5,000 sampled rows (estimated ~274 across all 137K rows). This is clearly a model failure case where the LLM produced a generic placeholder instead of the requested transformation. These rows predominantly appear in the indirect_reference prompt type. All such rows should be removed from the training set.

Other placeholder patterns found: "Lorem ipsum" (2 cases in original only), "TODO:" markers (1 in final, 2 in original), and the literal word "placeholder" (10 in final, 5 in original). These are relatively rare but should still be flagged for review.

## 4.1 Examples of Quick Brown Fox Fallback

Row 59 [indirect_reference]: final = "The quick brown fox jumps over the lazy dog.", original = "So we're back - with another interview for our Youth Week Writerly Types series!"

Row 1025 [indirect_reference]: final = "The quick brown fox jumps over the lazy dog.", original = "The websites www.24u.cz and www.24uSoftware.com..."

Row 1466 [indirect_reference]: final = "The quick brown fox jumps over the lazy dog.", original = "when i was younger my mother sat my siblings and i down..."

## 5. Truncation and Incomplete Text

Truncated or incomplete AI responses are problematic because they represent generation failures rather than valid AI text. A classifier trained on truncated AI text may learn to associate incomplete sentences with AI, which is not a robust feature for real-world detection.

### 5.1 Terminal Punctuation Analysis

9.61% of final_response entries (approximately 13,163 rows) do not end with terminal punctuation (period, exclamation, question mark, semicolon, colon, closing quote, bracket, or brace). While some of these are legitimate (e.g., ending with a URL, a product code, or a hashtag), many represent truncated outputs where the model's generation was cut off mid-sentence.

### 5.2 Examples of Non-Terminal Endings

|  Row | Prompt Type | Length | Last 30 chars  |
| --- | --- | --- | --- |
|  5 | indirect_reference | 5,558 | ...connection.\n\n---\n\n*End of Book*  |
|  34 | direct_reference | 4,130 | ...and allow us to contact you.*  |
|  39 | revise | 835 | ...#Borg #Nexus #TemporalAnomaly  |
|  44 | rewrite | 2,196 | ...operating temperature: 0 to 40 C  |
|  66 | indirect_reference | 1,385 | ...tic Proportional Control Valve  |

### 5.3 Suspicious Word Endings

Only 0.04% of rows end with a common function word (the, and, or, if, that, which, etc.) and no terminal punctuation. These are the most suspicious truncation cases. The most common ending words are "to" and "with", suggesting the model was mid-phrase when generation stopped. While the rate is very low, each case represents a clearly truncated output that should be reviewed.

## 6. Structural Markers and Label Leakage Risk

Label leakage occurs when the AI-generated text contains features that are trivially distinguishable from human text and would not appear in real-world AI output. A classifier trained on leaked features would appear accurate during evaluation but fail in deployment when real AI text doesn't contain those features. This is the most important quality dimension for a classifier training dataset.

### 6.1 Structural Feature Comparison

|  Feature | In final_response | In original | Leakage Risk  |
| --- | --- | --- | --- |
|  Bold markdown (**) | 35.0% | 0.7% | CRITICAL  |
|  Markdown headers (#) | 2.0% | 0.1% | HIGH  |
|  Bullet lists (-/') | 22.5% | 19.7% | Low  |
|  Numbered lists | 12.6% | 4.0% | MODERATE  |
|  Markdown links txt | 1.3% | 0.1% | HIGH  |
|  Code blocks (``) | 0.4% | 0.0% | MODERATE  |
|  Emoji | 3.4% | 1.1% | MODERATE  |

WARNING: Bold markdown formatting ( \( ^{**} \) ... \( ^{**} \) ) appears in 35% of AI texts but only 0.7% of human texts - a 50x ratio. This is the single largest label leakage risk in the dataset. The AI model (granite-4.1-8b) heavily favors markdown formatting in its outputs, particularly for headings, emphasis, and structure. A classifier could achieve >90% accuracy on this dataset simply by detecting markdown bold, which would not generalize to plain-text AI output.

WARNING: Markdown headers (# prefix), numbered lists, and markdown links are also significantly overrepresented in AI text. These are formatting preferences of the specific model used and would not appear if the generating model were changed or if output were post-processed to strip markdown.

Bullet lists are nearly equally common in both human and AI text (22.5% vs 19.7%), suggesting this is not a leakage risk. Emoji shows a moderate difference (3.4% vs 1.1%) that could contribute to leakage but is not dominant.

### 6.2 Explicit AI Self-Identification

OK: No explicit AI self-identification patterns were found at significant rates. Phrases like "As an AI", "I am an AI", and "as a language model" appear in fewer than 0.5% of AI texts and at similar rates in human texts (likely from texts about AI topics). The model does not appear to self-identify in its outputs, which is good for classifier quality.

### 6.3 Recommended Leakage Mitigations

Based on the analysis above, the following post-processing steps are strongly recommended before using this dataset for classifier training:

1. Strip markdown formatting: Remove **bold**, ## headers, links, and ``code blocks`` from all final_response entries. This eliminates the largest leakage feature.
2. Remove meta-commentary: Filter or rewrite sentences containing "In summary", "In conclusion", "the original text", "the above passage", and similar AI artifacts that reference the transformation task.
3. Remove placeholder outputs: Drop all rows where final_response is "The quick brown fox jumps over the lazy dog" or similar fallback text.
4. Length normalization: Consider training with length-balanced batches or adding length as a controlled feature to prevent the classifier from relying on length as a primary signal.

## 7. Per-Prompt-Type Quality Analysis

The dataset uses four prompt types to generate diverse AI text. Each type applies a different manipulation strategy to the original text, producing AI text with different characteristics. Understanding per-type quality is essential for ensuring balanced training data.

### 7.1 Prompt Type Definitions

|  Prompt Type | Count | Description  |
| --- | --- | --- |
|  direct_reference | 34,272 (25.0%) | AI text directly references or cites the original  |
|  revise | 34,267 (25.0%) | AI revises or edits the original text  |
|  indirect_reference | 34,260 (25.0%) | AI writes about the same topic without direct reference  |
|  rewrite | 34,234 (25.0%) | AI rewrites the text with different wording/style  |

### 7.2 Per-Type Quality Metrics

|  Metric | direct_reference | revise | indirect_reference | rewrite  |
| --- | --- | --- | --- | --- |
|  Mean orig length | 3,380 | 3,289 | 3,238 | 3,134  |
|  Mean final length | 2,545 | 2,801 | 2,419 | 2,326  |
|  Empty final | 0 (0.0%) | 1 (0.1%) | 0 (0.0%) | 0 (0.0%)  |
|  Identical to orig | 1 (0.1%) | 0 (0.0%) | 0 (0.0%) | 11 (0.9%)  |
|  No end punctuation | 5.8% | 17.2% | 8.3% | 9.5%  |
|  Very short final | 0 (0.0%) | 1 (0.1%) | 7 (0.6%) | 1 (0.1%)  |
|  AI preamble | 0 (0.0%) | 8 (0.7%) | 2 (0.2%) | 0 (0.0%)  |

WARNING: The revise prompt type has significantly higher no-terminal-punctuation rate (17.2%) compared to other types (5.8-9.5%). This suggests the revise prompt tends to produce longer, more structured outputs that get truncated or end with formatting artifacts. This type needs the most post-processing attention.

WARNING: The indirect_reference type accounts for all "quick brown fox" placeholder cases and has the highest short-final rate (0.6%). This suggests the model sometimes fails to produce meaningful content when asked to write about a topic without directly referencing the source text.

The rewrite type has the highest identical-to-original rate (0.9%), meaning 11 out of 1,276 sampled rows produced no changes. This could indicate model laziness or cases where the original text was already so well-written that the model decided no changes were needed.

### 7.3 Prompt Template Diversity

86 unique prompt templates were found in the 5,000-row sample. The top templates cover diverse manipulation strategies including: taking inspiration from style, writing prequels/followups, writing contradictory text, writing on the same topic, writing with citations, writing alternate versions, expanding subpoints, writing rebuttals, writing on different topics, and translation (German, Spanish, Hindi). This diversity is good for training a classifier that must detect AI text across many generation strategies.

The most common template (3.9%) simply inserts the document and asks to "take inspiration from the style, language, and content of this text, and write a new piece." The top 10 templates collectively cover about 23% of the data, with the remaining 77% spread across 76+ other templates. This long-tail distribution ensures good coverage of diverse generation strategies.

## 8. Response Selection Analysis

Each row can have up to two candidate AI responses (response_0 and response_1). The final_response is selected from one of these candidates. Understanding the selection pattern is important for data quality: if the selection is biased toward certain features, it could amplify leakage risks.

### 8.1 Selection Distribution

|  Selection | Count | Percentage  |
| --- | --- | --- |
|  final_response == response_0 | 68,435 | 49.94%  |
|  final_response == response_1 | 68,701 | 50.13%  |
|  final_response != both (modified) | 1,817 | 1.33%  |

OK: The selection is nearly perfectly balanced between response_0 and responseB response_1 (50/50), indicating no systematic selection bias. This is good for training quality.

WARNING: 1,817 rows (1.33%) have final_response that matches neither response_0 nor response_1. These represent post-generation modifications. Examples show cases where the response was stripped of AI preamble (e.g., removing "Sure, here's a more conversational version:" from the beginning) or where additional formatting was applied. This post-processing is actually a good sign that some boilerplate removal was applied, but the coverage may be incomplete.

### 8.2 Identical to Original

238 rows (0.17%) have final_response identical to original. These provide no training value for a classifier since the AI and human texts are the same, making the class label ambiguous. These should be removed from training.

## 9. Encoding and Character Issues

Encoding issues can corrupt text and introduce artifacts that confuse a classifier. The dataset was checked for BOM markers, replacement characters, mojibake, control characters, and extremely long single-line texts.

|  Issue | Count (in 5K sample) | Severity  |
| --- | --- | --- |
|  Replacement character (\ufffd) | 5 | Low - scattered encoding failures  |
|  Very long single-line text (>5K chars, no newlines) | 7 | Low - unusual formatting  |
|  BOM markers | 0 | None found  |
|  Mojibake sequences (3+ consecutive) | 0 | None found  |
|  Control characters (\x00-\x1f) | 0 | None found  |

OK: The dataset is remarkably clean from an encoding perspective. No BOM markers, mojibake, or control characters were found. The 5 replacement characters and 7 long single-line texts are negligible. The dataset creator's filtering pass appears to have been effective for encoding cleanup.

### 9.1 Language Consistency

Non-Latin characters (Cyrillic, Arabic, Devanagari, CJK, Korean) appear in 0.1% of AI final_response and 0.9% of human original text. Importantly, no language mismatch was found: there are zero cases where an English original produced a non-Latin AI response. Some human originals contain non-Latin text (e.g., Hindi, CJK), and the AI model correctly preserves or translates these. This is good for data quality.

# 10. Cross-Shard Consistency and Dataset-Level Concerns

Cross-shard consistency could not be verified via streaming due to download constraints, but several dataset-level concerns can be identified from the shard_0 analysis.

## 10.1 Single Model Risk

WARNING: The entire dataset is generated by a single model: cyankiwi/granite-4.1-8b-AWQ-INT4 (a quantized 8B parameter model). This creates a significant generalization risk: a classifier trained on this data will learn to detect text from THIS specific model, not AI text in general. Different models (GPT-4, Claude, Llama, Mistral, etc.) have different writing styles, formatting preferences, and artifact patterns. A classifier that achieves 99% accuracy on this dataset may drop to 60-70% on text from other models.

To mitigate this, consider: (1) mixing data from multiple generator models, (2) augmenting with AI text from different model families and sizes, (3) evaluating the classifier on out-of-distribution AI text during testing.

## 10.2 Single Generation Config

All rows use the same generation parameters: temperature=0.6, top_p=0.9, presence_penalty=1.5, top_k=40. While this ensures consistency, it also limits diversity. Higher temperature settings would produce more varied AI text, while lower settings would produce more deterministic outputs. A production classifier should be robust across generation settings.

## 10.3 Duplicate Originals

Hash-based duplicate detection found that some original texts appear more than once (paired with different AI responses). While this is not necessarily a problem - the same human text can legitimately have multiple AI variants - it could cause train/test leakage if the same original appears in both sets. Ensure that splits are made at the original-text level, not the row level.

## 10.4 Missing Shard 2

Shard 2 was unavailable at the time of analysis (still generating). This means ~137K rows are missing from the dataset. If shard 2 contains different prompt types or generation parameters, the current dataset may be unrepresentative of the full intended distribution.

## 11. Summary of Findings and Recommendations

### 11.1 Critical Issues (Must Fix)

|  # | Issue | Impact | Recommendation  |
| --- | --- | --- | --- |
|  1 | Markdown formatting leakage (35% bold, 2% classifier) | learns model formatting, not AI text | Markdown from final_response before training  |
|  2 | Summary meta-commentary (5.5% of AI text) | Trivial AI signal that won't generalize | Remove or rewrite sentences with "In summary/In conclusion/To  |
|  3 | Original text references (1.4-1.8% of AI text) | Artifact of rewrite task, not AI nature | Remove phrases referencing "the original text/meaning/passage  |
|  4 | Quick brown fox placeholder (~0.2% of rows) | No useful training signal | Drop these rows entirely  |
|  5 | Single generator model | Overfitting to one model's style | Augment with multiple model families  |

### 11.2 Important Issues (Should Fix)

|  # | Issue | Impact | Recommendation  |
| --- | --- | --- | --- |
|  6 | Length asymmetry (AI 25% shorter) | Classifier may learn length proxy | Length-balanced training or add length as controlled feature  |
|  7 | 14 empty + 24 very short final_response | No training value | Remove these rows  |
|  8 | 238 identical final_response==original | Ambiguous class label | Remove these rows  |
|  9 | No terminal punctuation (9.6%) | Some are truncated outputs | Review and possibly remove clear truncation cases  |
|  10 | Revise type 17.2% no-end-punct rate | Systematic truncation for one prompt | Reprocess revise outputs for terminal punctuation  |

### 11.3 Minor Issues (Nice to Fix)

|  # | Issue | Impact | Recommendation  |
| --- | --- | --- | --- |
|  11 | use_multiturn flag always False | Metadata inconsistency | Fix flag to match actual turn count  |
|  12 | 1,817 final_response != r0 or r1 | Unknown post-processing | Document the selection/editing logic  |
|  13 | 5 replacement characters | Encoding artifacts | Remove or fix affected rows  |
|  14 | Code blocks in AI text (0.4%) | Model-specific formatting | Strip as part of markdown cleanup  |
|  15 | Single generation config | Limited diversity | Consider varying temperature/top_p  |

### 11.4 Estimated Data Loss from Cleaning

If all recommended row-level removals are applied (empty/short final_response, identical pairs, placeholder text, and encoding issues), the estimated data loss is approximately 0.5% of rows (under 700 out of 137,033). The post-processing mitigations (markdown stripping, meta-commentary removal) would be applied in-place and would not remove any rows. This means the dataset remains large and diverse even after cleaning, which is an excellent starting position for building a robust classifier.

Overall Assessment: The dataset is well-structured and mostly clean, with strong prompt diversity and balanced type distribution. The primary risk is label leakage from model-specific formatting (markdown) and task-specific meta-commentary. These are addressable through straightforward post-processing. The single-model limitation is the most fundamental concern and should be addressed by augmenting with additional model outputs before final classifier training.