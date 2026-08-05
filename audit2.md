# Dataset Quality Audit Report

G-reen/cc-2021-rewritten

Comprehensive Inspection of 23,716 Rows Across 7 Shards

Generated: 2026-08-03

## Executive Summary

This report presents a comprehensive quality audit of the G-reen/cc-2021-rewritten dataset, which is designed as a test set for training and evaluating AI text classifiers. The dataset contains 23,716 rows distributed across 7 shards (shard_0 through shard_6), each containing approximately 3,350-3,430 examples. Each row pairs a human-written original text with an AI-generated response, along with metadata about the generation process including the prompt type, model used, and generation parameters.

The audit identified 18 distinct issue categories, ranging from empty responses and model refusals to boilerplate contamination, encoding anomalies, and structural inconsistencies. While the dataset is generally well-structured with consistent schemas across shards, several systematic issues were discovered that could compromise classifier training and evaluation if left unaddressed. The most impactful findings include: 720 rows where response_0 contains a regenerated prompt instead of actual AI output (3.04%), 517 rows where final_response matches neither response_0 nor response_1 (2.18%), and 88 refusal-like responses where the model declined to generate text (0.37%). Additionally, approximately 7.7% of a sampled subset showed near-identical original and AI texts (Jaccard similarity > 0.9), suggesting trivial or ineffective transformations that would provide poor training signal for a classifier.

### Executive Summary 1

#### 1. Dataset Overview 3

1.1 Schema and Structure 3
1.2 Prompt Types 3
1.3 Generator Models 3
1.4 Response Field Relationship 4

#### 2. Issue Findings 5

2.1 Empty and Missing Responses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.2 Model Refusals and Inability Responses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.3 Prompt Leakage in response_0 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.4 final_response Mismatch with Source Responses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
2.5 Remaining Boilerplate Contamination . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
2.6 Suspiciously Short Responses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
**3. Structural and Content Quality Issues** . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
3.1 Near-Identical Original and AI Pairs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
3.2 Length Ratio Anomalies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
3.3 Encoding and Character Anomalies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
3.4 Repetition and Degenerate Outputs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
3.5 Truncation and Incomplete Responses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
3.6 Emoji Injection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
3.7 Cookie Notice and Boilerplate Originals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
**4. Issue Distribution Analysis** . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.1 Per-Model Issue Rates . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.2 Per-Prompt-Type Issue Rates . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4.3 "Okay" Prefix Deep Dive . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
**5. Summary of All Issues** . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
**6. Recommendations** . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
6.1 Critical Fixes (Before Training) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
6.2 High-Priority Cleaning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
6.3 Medium-Priority Improvements . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
6.4 Data Quality Flags for Downstream Use . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

## 1. Dataset Overview

### 1.1 Schema and Structure

The dataset uses a consistent schema across all 7 shards. Each row contains the following columns: original (the human-written source text), prompt (a structured object containing chat turns, metadata about the prompt type, and flags), response_0 (the first model response), response_1 (the second model response, empty for single-turn prompts), final_response (the selected response to pair with the original), generator_model (the model used), and generation_params (JSON string of generation parameters). The prompt object contains a metadata field with PROMPT_TYPE, a use_multiturn boolean flag (always False), chat_turns (a list of strings, 1 or 2 turns), and an examples list.

|  Shard | Rows | Model | Empty final_resp | Empty resp_0 | Empty resp_1  |
| --- | --- | --- | --- | --- | --- |
|  shard_0 | 3,395 | cyankiwi/granite-4.1-8b-AWQ-INT4 | 1 | 0 | 1,732  |
|  shard_1 | 3,378 | Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | 0 | 0 | 1,725  |
|  shard_2 | 3,410 | Qwen/Qwen3-8B-AWQ | 4 | 8 | 1,740  |
|  shard_3 | 3,358 | google/gemma-4-E4B-it | 0 | 0 | 1,718  |
|  shard_4 | 3,371 | cyankiwi/Ministral-3-8B-Instruct-AWQ-4bit | 0 | 0 | 1,722  |
|  shard_5 | 3,373 | Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | 0 | 0 | 1,723  |
|  shard_6 | 3,431 | cyankiwi/granite-4.1-8b-AWQ-INT4 | 0 | 0 | 1,744  |

### 1.2 Prompt Types

The dataset employs four distinct prompt strategies to generate diverse AI text variants from each original. These prompt types are distributed roughly equally across the dataset, ensuring balanced representation. The direct_reference type (5,866 rows, 24.73%) provides the original text directly and asks the model to write a new piece inspired by its style, often as a "rough draft typed on a phone." The indirect_reference type (5,923 rows, 24.97%) uses a two-turn approach: first generating a style descriptor from the original, then asking a model to recreate text from that descriptor. The revise type (6,002 rows, 25.31%) asks the model to change the tone of the original text (e.g., to be more relaxed and friendly). The rewrite type (5,925 rows, 24.98%) asks the model to modify the text to be as long as possible while retaining at least half of the original text verbatim.

### 1.3 Generator Models

Five distinct quantized models were used to generate AI responses. Notably, each shard is associated with a single model (except that Meta-Llama-3.1-8B-Instruct-AWQ-INT4 and

granite-4.1-8b-AWQ-INT4 each appear in two shards). The model distribution is: cyankiwi/granite-4.1-8b-AWQ-INT4 (6,826 rows, 28.78%), hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 (6,751 rows, 28.47%), Qwen/Qwen3-8B-AWQ (3,410 rows, 14.38%), cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit (3,371 rows, 14.21%), and google/gemma-4-E4B-it (3,358 rows, 14.16%). All models used consistent generation parameters with temperature ranging from 0.6 to 1.25, top_p from 0.8 to 1.0, and a presence_penalty of 1.5 across the board.

### 1.4 Response Field Relationship

A key structural finding concerns how final_response relates to response_0 and response_1. For rows where response_1 is empty (12,104 rows, the single-turn cases), final_response equals response_0 in 97.43% of cases. For rows where response_1 is non-empty (11,612 rows, the multi-turn cases), final_response equals response_1 in 98.19% of cases. This indicates that final_response is typically selected from the "best" response: response_0 for single-turn, response_1 for multi-turn. However, there are 517 rows (2.18%) where final_response differs from both response_0 and response_1, suggesting a post-processing or filtering step that further cleans the selected response. Of these 517 mismatches, 516 are genuinely different text (not just whitespace differences), indicating that this filtering strips prefixes, headings, or other boilerplate from the selected response to produce a cleaner final_response.

## 2. Issue Findings

### 2.1 Empty and Missing Responses

Empty response fields represent the most straightforward data quality issue: the model failed to produce any output for these rows. The audit found 5 rows with empty final_response, 8 rows with empty response_0, and 12,104 rows with empty response_1. The response_1 empties are expected and structural (single-turn prompts only generate one response), so they do not represent errors. However, the 5 empty final_response and 8 empty response_0 are genuine failures that should be addressed.

Examining the empty final_response rows reveals distinct failure modes. Row 2,455 (shard 0, granite model) shows response_0 containing a regenerated prompt rather than actual output. Row 8,530 (shard 2, Qwen) has both response_0 and response_1 empty, indicating a complete generation failure. Row 8,584 (shard 2, Qwen) shows response_0 verbatim copying the original text without transformation. Row 8,736 (shard 2, Qwen) produced a degenerate response of repeated "A ..." patterns. These failures are concentrated in Qwen/Qwen3-8B-AWQ (shard 2), which accounts for 4 of the 5 empty final_response rows and all 8 empty response_0 rows, suggesting this model may have reliability issues with certain prompt types.

### 2.2 Model Refusals and Inability Responses

A total of 88 rows (0.37%) contain refusal-like patterns in final_response, where the model either explicitly declined to generate text or indicated it could not complete the request. These break down into several categories: "I cannot fulfill/complete/do..." (14 rows), "I am unable to..." (20 rows), "I will not..." (32 rows, though many of these are legitimate text content rather than actual refusals), "As an AI..." (8 rows), and requests for source text such as "Please provide the source text" (11 rows). The most concerning are the template placeholders like "[Original Passage Placeholder]" and explicit inability statements such as "I'm unable to generate the requested text because you haven't provided the original passage." These responses provide no useful training signal for a classifier since they are clearly AI-generated boilerplate that does not resemble the original text at all.

The "I will not" pattern (32 rows) requires careful examination because many instances are not actual refusals but rather legitimate text content containing that phrase. Examples include "I will not be applying for the permanent position" (a job-related text), "I will not let negative, jealous people think that I can't do it" (motivational content), and "I will not reason and compare; my business is to obey" (a literary quotation). Only a small subset represent genuine model refusals, such as "I will not provide additional support for this initiative." This distinction is important: blindly filtering all "I will not" matches would remove valid training data.

### 2.3 Prompt Leakage in response_0

One of the most significant findings is that 720 rows (3.04%) have response_0 that begins with an instruction verb (Generate, Rewrite, Revise, Please, Create, Write) and differs from final_response. In these cases, the model regenerated the prompt or produced instruction-style text instead of actually completing the requested task. This is overwhelmingly concentrated in the indirect_reference prompt type (716 of 720 rows, 99.4%), which uses a

two-turn conversation structure. The leakage is distributed across models: Meta-Llama-3.1-8B (364 rows), Qwen3-8B (165), gemma-4-E4B-it (163), granite-4.1-8b (27), and Ministral-3-8B (1).

This pattern suggests that in the indirect_reference workflow, the first model turn (which generates a style descriptor from the original text) sometimes produces a full prompt regeneration instead of the expected descriptor. For example, instead of writing "This text uses formal academic language with an average sentence length of 25 words...", the model outputs "Generate a text that adheres to the following constraints: 1. Format: The text must be structured as..." This is a systematic issue with the indirect_reference prompt design that causes models to misinterpret the task as a prompt-writing exercise rather than a description task. The good news is that final_response correctly uses response_1 (the second turn's output) in most of these cases, so the actual paired AI text is typically still valid. However, these rows should be flagged because the response_0 field contains garbage data that should not be used for any analysis.

### 2.4 final_response Mismatch with Source Responses

There are 517 rows (2.18%) where final_response differs from both response_0 and response_1. Of these, 516 are genuinely different text (not merely whitespace variants). Inspection reveals this is a deliberate post-processing step that strips boilerplate prefixes and structural additions from the raw model output. For example, response_0 might contain "**Course Overview: Mastering Digital Marketing for Small Businesses**\n\n---\n\n**What's Included?**" while final_response contains only "**What's Included?**" with the title and horizontal rule removed. Similarly, response_0 might start with "Sure! Here's a more relaxed and friendly version of the text:\n\n---\n\n" while final_response strips this preamble entirely. This is actually a beneficial filtering step that improves data quality. However, it raises the question of whether this filtering is comprehensive enough, as the audit still found remaining boilerplate in final_response (see Section 2.5).

### 2.5 Remaining Boilerplate Contamination

Despite the initial filtering pass mentioned in the dataset description, significant boilerplate contamination remains in final_response. The audit detected multiple categories of residual boilerplate:

|  Category | Count | Percentage | Severity  |
| --- | --- | --- | --- |
|  "Okay" conversational prefix | 97 | 0.41% | MEDIUM  |
|  "Here's/Here is" prefix | 41 | 0.17% | MEDIUM  |
|  "Sure" prefix | 13 | 0.05% | LOW  |
|  "If you have any questions" suffix | 151 | 0.64% | MEDIUM  |
|  "Please let me know" suffix | 18 | 0.08% | LOW  |
|  "Feel free to ask" suffix | 6 | 0.03% | LOW  |
|  Markdown heading/title prefix | 245 | 1.03% | HIGH  |
|  Markdown code block wrapping | 13 | 0.05% | LOW  |
|  Template placeholder ([Insert...]) | 110 | 0.46% | HIGH  |
|  XXX filler text | 76 | 0.32% | HIGH  |

The "Okay" prefix category is notable because many of these are legitimate outputs from the direct_reference prompt type, which asks the model to write "as if it was a rough draft being typed out on a phone." In this context, starting with "okay so i'm trying to write this thing about..." is an intentional stylistic choice, not boilerplate. However, for classifier training purposes, these conversational openings are strong AI signals that may cause the classifier to learn superficial patterns rather than deeper stylistic differences. The template placeholder issues (110 "[Insert..." patterns and 76 "XXX" fillers) are more concerning as they indicate the model failed to complete the task properly and left structural placeholders in the output.

## 2.6 Suspiciously Short Responses

A total of 15 rows have final_response shorter than 50 characters, which is suspiciously short given that original texts average 3,388 characters. These short responses fall into several categories: genuine model failures such as "I cannot fulfill this request." (Row 10,848) and "I can't proceed with that request." (Row 17,814), template placeholders like "[Insert Source Text Here]" (Row 10,558) and "[Source Text to be provided here]" (Row 13,310), degenerate outputs like "Down BEARISH" (Row 20,466, from a 695-character original about financial markets), and legitimate but very short transformations like "Maggie, I love you. Van, I see you again." (Row 1,177, from a 738-character original). The degenerate and failure cases should be removed, while the legitimate short transformations should be reviewed individually to determine if they provide useful training signal.

## 3. Structural and Content Quality Issues

### 3.1 Near-Identical Original and AI Pairs

In a 5,000-row sample, 386 rows (7.72%) showed Jaccard similarity greater than 0.9 between original and final_response, indicating near-identical text. An additional 474 rows (9.48%) fell in the 0.7-0.9 similarity range. For a classifier training dataset, near-identical pairs are problematic because they provide minimal discriminative signal: if the human and AI texts are nearly the same, the classifier cannot learn meaningful distinguishing features. Extrapolating to the full dataset, this suggests approximately 1,800-2,000 rows may have overly similar pairs. Additionally, 76 rows have exact string matches between original and final_response, meaning the AI produced the identical text as the human original with no transformation at all.

The near-identical cases are likely concentrated in specific prompt types. The revise prompt type, which asks for tone changes, may produce very similar text when the original is already in the target tone. The rewrite prompt type, which requires retaining at least half the original verbatim, naturally produces higher similarity. These near-duplicates should be filtered or downweighted during training to prevent the classifier from learning that similar texts can be either human or AI (which would encourage decision boundary uncertainty).

### 3.2 Length Ratio Anomalies

The length ratio between final_response and original text reveals significant outliers. On average, AI responses are similar in length to originals (mean ratio 1.185, median 0.976), but the distribution has heavy tails. 217 rows (0.91%) have final_response less than 10% of the original length, indicating the model produced a drastically abbreviated output or failed mid-generation. Conversely, 372 rows (1.57%) have final_response more than 5x the original length, with the most extreme case reaching 7.8x (a 14,775-char original generating a 115,277-char response). These extreme length mismatches could bias a classifier toward using length as a primary feature rather than learning deeper stylistic patterns.

The short-ratio cases include model failures ("I'm unable to proceed with the task..."), legitimate but overly condensed summaries (a 9,734-char original about Viagra reduced to a 662-char medical description), and truncation artifacts. The long-ratio cases typically involve the rewrite prompt type asking for maximum length expansion, but some represent runaway generation where the model repeated content or added excessive detail. Both extremes should be reviewed for inclusion in the training set.

### 3.3 Encoding and Character Anomalies

Encoding issues were detected in 305 rows (1.29%) containing mojibake characters (e.g., "i", "i", "½") and 29 rows (0.12%) containing Unicode replacement characters (☒), which indicate corrupted or improperly decoded text. These issues appear in both original (172 mojibake, 26 replacement chars) and final_response (133 mojibake, 3 replacement chars) fields. The higher rate in original texts suggests the source data (likely web-scraped content) contained encoding issues that were partially but not fully cleaned. Additionally, 4 rows in original and 11 rows in final_response have non-ASCII character ratios exceeding 30%, which may indicate non-English content or specialized character sets. Two original texts and one final_response contain literal "\n" escape sequences instead of actual newline characters, suggesting a

double-escaping issue in the data pipeline.

### 3.4 Repetition and Degenerate Outputs

A total of 44 rows (0.19%) contain detectable repetition patterns in final_response, where a chunk of at least 20 characters is repeated 3 or more times consecutively. Examples include repeated whitespace indentation patterns, horizontal rules ("---" repeated 4 times), and underscore lines ("____" repeated 3 times). While some repetition is legitimate (e.g., markdown tables with repeated divider lines), others indicate generation artifacts. One notable case (Row 8,736) produced "A ... \nA ... \nA ..." repeated many times, a clearly degenerate output. These repetition patterns, while infrequent, are strong AI signatures that could cause a classifier to overfit on this particular artifact rather than learning generalizable features.

### 3.5 Truncation and Incomplete Responses

A significant 3,906 rows (16.47%) have final_response that does not end with standard sentence-ending punctuation (period, exclamation mark, question mark, ellipsis, quote, or parenthesis). While some of these are legitimate (text ending with a URL, a code snippet, or a list item), many likely represent truncated outputs where the model hit a token limit or the response was cut off during generation. Specifically, 384 rows end with ellipsis (...), 8 end with dashes (--), 36 contain "[cont" continuation markers, and 2 contain "to be continued" phrases. The high rate of potentially truncated responses (16.47%) is concerning because truncated AI text may have different statistical properties than complete text, potentially introducing noise into classifier training. A classifier might learn that incomplete sentences are an AI signal, which would not generalize well to real-world AI text that is not truncated.

### 3.6 Emoji Injection

In 673 rows (2.84%), the AI model added emojis that were not present in the original human text. This emoji injection is heavily concentrated in the direct_reference (282 rows) and revise (244 rows) prompt types, with fewer cases in indirect_reference (131) and rewrite (16). Common injected emojis include conversation openers like wave and sparkles symbols, topic-relevant emojis (paw prints for pet content, globe for travel), and decorative markers (stars, checkmarks). This is a very strong AI signal that could cause a classifier to over-rely on emoji presence as a discriminating feature. If the classifier encounters AI-generated text without emojis at inference time (e.g., from a model that does not use emojis), its accuracy would drop. Consider either stripping emojis from both human and AI text before training, or ensuring the training distribution includes sufficient AI text without emojis to prevent this overfitting.

### 3.7 Cookie Notice and Boilerplate Originals

A total of 707 rows (2.98%) contain cookie/privacy/policy content in the original text, and 72 rows (0.30%) have exact duplicate originals (same text appearing multiple times). The top duplicated original is a cookie consent notice ("This website uses cookies to improve your experience while you navigate through the website...") appearing 20 times. Another variant appears 15 times, and a slightly different version appears 8 times. These duplicates arise because the source dataset (likely Common Crawl derived) includes many websites with

identical cookie notices. While these rows are not "errors" per se, they reduce the effective diversity of the dataset and could cause the classifier to overfit on cookie notice language as either a human or AI signal. The duplicated originals also mean the AI responses for these rows are all different transformations of the same source text, which could create an unintended weighting effect during training.

## 4. Issue Distribution Analysis

### 4.1 Per-Model Issue Rates

Issue rates vary significantly across generator models, revealing model-specific failure patterns. The table below summarizes key issue rates per model. Qwen/Qwen3-8B-AWQ stands out with the highest rates of empty responses (0.12% final_response, 0.23% response_0) and the highest boilerplate prefix rate (1.23%). cyankiwi/granite-4.1-8b-AWQ-INT4 has the highest refusal rate (0.38%) and the second-highest boilerplate rate (0.37%). cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit has the highest template/placeholder leakage rate (1.01%) but zero refusals and zero empty responses, suggesting it reliably produces output but sometimes fails to complete the task properly. google/gemma-4-E4B-it has the highest short-response rate (0.24%) and the highest boilerplate rate (0.83%), while Meta-Llama-3.1-8B-Instruct-AWQ-INT4 has moderate rates across all categories.

|  Model | Empty FR | Short FR | Refusals | Boilerplate | Template Leak  |
| --- | --- | --- | --- | --- | --- |
|  Qwen3-8B-AWQ | 0.12% | 0.00% | 0.15% | 1.23% | 0.09%  |
|  granite-4.1-8b | 0.01% | 0.06% | 0.38% | 0.37% | 0.21%  |
|  Ministral-3-8B | 0.00% | 0.00% | 0.00% | 0.03% | 1.01%  |
|  gemma-4-E4B-it | 0.00% | 0.24% | 0.15% | 0.83% | 0.74%  |
|  Llama-3.1-8B | 0.00% | 0.04% | 0.07% | 0.59% | 0.53%  |

### 4.2 Per-Prompt-Type Issue Rates

Issue patterns also vary by prompt type. The indirect_reference type has the highest rate of refusal-like responses (24 rows, 0.41%) and suspiciously short final_responses (13 rows, 0.22%), likely because its two-turn structure introduces more failure points. It also accounts for virtually all of the prompt leakage in response_0 (716 of 720 rows). The direct_reference type has the highest boilerplate prefix rate (114 rows, 1.94%), consistent with its prompt asking for a casual "rough draft typed on a phone" style that encourages conversational openings like "Okay so..." and "Hey there!". The rewrite type has moderate refusal rates (10 rows, 0.17%) and the longest average final_response length (3,545 chars vs. 3,448 original), as expected from its "make it as long as possible" instruction. The revise type is the cleanest, with zero short responses and only 6 boilerplate prefixes, reflecting its simpler "change the tone" instruction that models handle reliably.

### 4.3 "Okay" Prefix Deep Dive

The "Okay" prefix in final_response (97 rows) deserves special analysis because it straddles the boundary between legitimate stylistic output and boilerplate contamination. Of these 97 rows, 88 are from the direct_reference prompt type, which explicitly asks the model to write "as if it was a rough draft being typed out on a phone." In this context, starting with "okay so i'm trying to write this thing about..." is an intentional stylistic emulation of casual human drafting behavior. However, this creates an ambiguity for classifier training: the AI text successfully mimics a human casual writing style, but the "okay" opener is a strong statistical signal that a classifier could exploit in a way that doesn't generalize to other AI generation methods. The model distribution for "Okay" prefixes is Qwen3-8B (39), Llama-3.1-8B (33), gemma-4-E4B-it

(22), and granite-4.1-8b (2).

## 5. Summary of All Issues

The table below provides a consolidated view of all identified issues, ordered by estimated severity and impact on classifier training. Severity ratings consider both the frequency of the issue and its potential to introduce noise or bias into a classifier trained on this data.

|  # | Issue | Affected Rows | Rate | Severity  |
| --- | --- | --- | --- | --- |
|  1 | Empty response_1 (single-turn, expected) | 12,104 | 51.04% | INFO  |
|  2 | Potentially truncated responses | 3,906 | 16.47% | MEDIUM  |
|  3 | Near-identical orig vs AI (Jaccard > 0.9) | ~1,830 est. | ~7.7% | HIGH  |
|  4 | Prompt leakage in response_0 | 720 | 3.04% | HIGH  |
|  5 | Emoji injected by AI (not in original) | 673 | 2.84% | MEDIUM  |
|  6 | final_response mismatch (post-processing) | 517 | 2.18% | INFO  |
|  7 | AI response > 5x original length | 372 | 1.57% | MEDIUM  |
|  8 | Mojibake/encoding issues | 305 | 1.29% | MEDIUM  |
|  9 | AI response < 10% original length | 217 | 0.91% | HIGH  |
|  10 | Template placeholder leakage ([Insert], XXX) | 186 | 0.78% | HIGH  |
|  11 | Suffix boilerplate ("If you have questions...") | 182 | 0.77% | MEDIUM  |
|  12 | Boilerplate prefix (Sure/Okay/Here's) | 151 | 0.64% | MEDIUM  |
|  13 | Markdown heading prefix (---) | 245 | 1.03% | MEDIUM  |
|  14 | Cookie/boilerplate duplicated originals | 72 | 0.30% | LOW  |
|  15 | Refusal-like responses | 88 | 0.37% | HIGH  |
|  16 | Repetition in final_response | 44 | 0.19% | LOW  |
|  17 | Unicode replacement characters | 29 | 0.12% | LOW  |
|  18 | Suspiciously short final_response (<50 chars) | 15 | 0.06% | HIGH  |

## 6. Recommendations

### 6.1 Critical Fixes (Before Training)

Remove empty and degenerate responses: The 5 rows with empty final_response and the 15 rows with suspiciously short responses (<50 chars) should be removed entirely. These provide

no useful training signal and would introduce noise. The 8 rows with empty response_0 in shard_2 should also be flagged and reviewed, as they indicate Qwen model failures that may affect other fields in those rows.

Remove refusal responses: The 88 rows with refusal-like patterns should be removed from the training set. These are clearly AI-generated boilerplate ("I cannot fulfill this request", "Please provide the source text") that does not represent the type of AI text the classifier will encounter at inference time. Keeping these would teach the classifier to detect "I cannot" phrases rather than genuine AI stylistic features.

Remove template placeholder responses: The 186 rows containing [Insert...], [Source Text...], or XXX filler should be removed. These represent incomplete generation where the model failed to fill in the required content, producing structurally correct but semantically empty text.

### 6.2 High-Priority Cleaning

Filter near-identical pairs: Rows where Jaccard similarity between original and final_response exceeds 0.9 should be either removed or downweighted during training. The estimated ~1,830 near-identical rows provide minimal discriminative signal and could cause the classifier to learn that similar texts can be either label, increasing decision boundary uncertainty. Consider keeping a small subset for robustness but removing the bulk.

Strip remaining boilerplate: Extend the existing filtering pass to catch the identified residual patterns: conversational prefixes (Sure, Okay, Here's, Certainly), courtesy suffixes (If you have any questions, Please let me know, Feel free to ask), and markdown structural prefixes (headings, horizontal rules at the start). Be careful with the "Okay" prefix from direct_reference prompts, as these may be intentional stylistic outputs rather than boilerplate. Consider stripping only if the prefix is followed by a line break or markdown structural element, indicating it's a preamble rather than part of the text flow.

Handle emoji injection: Either strip all emojis from both human and AI text before training to eliminate this confounding signal, or augment the training data with AI-generated text that does not use emojis to prevent the classifier from over-relying on emoji presence as an AI indicator.

### 6.3 Medium-Priority Improvements

Address length ratio extremes: Review the 217 rows where final_response is less than 10% of the original length and the 372 rows where it exceeds 5x. Remove or flag entries where the extreme ratio indicates a generation failure rather than an intentional transformation. Keep entries where the ratio is a natural consequence of the prompt type (e.g., rewrite asking for longer text).

Fix encoding issues: The 305 rows with mojibake and 29 rows with Unicode replacement characters should have their text re-encoded or cleaned. For mojibake, attempt UTF-8 re-decoding of the raw bytes. For replacement characters, the original text is likely unrecoverable and these rows should be flagged as lower quality.

Deduplicate cookie notice originals: The 72 exact-duplicate originals (primarily cookie consent notices) should be deduplicated or downweighted to prevent overrepresentation of boilerplate legal/privacy text in the training distribution.

### 6.4 Data Quality Flags for Downstream Use

Several issues are informational rather than actionable but should be documented for downstream users. The 517 rows where final_response differs from both response_0 and response_1 indicate a beneficial post-processing step; this should be documented in the dataset card. The 720 rows with prompt leakage in response_0 should be flagged so that users know not to use response_0 for analysis in those cases. The 16.47% truncation rate should be documented as a known limitation, and users should be aware that truncated AI text may have different statistical properties than complete text. Finally, the per-model quality differences (especially Qwen's higher failure rate) should be noted so that users can make informed decisions about shard selection or model-weighted sampling during training.