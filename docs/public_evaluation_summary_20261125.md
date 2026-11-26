# DocuAgent — Public Evaluation Summary

Documentation date: **25 November 2026**. This document summarizes existing experiments; it does not report new model calls or a deployment benchmark.

## Scope and evidence types

DocuAgent is a local single-user PDF question-answering prototype. The application uses scoped single-pass retrieval, a deterministic LangGraph workflow and technical source-reference validation. Semantic search is the default; hybrid modes are optional configurations.

The evaluations distinguish:

- **Controlled fixtures:** fixed vectors or intentionally good/faulty answers used to test behavior and calculations.
- **Local embeddings:** rankings computed with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` over synthetic documents.
- **Stored real model responses:** actual GPT-4.1-mini answers generated from frozen synthetic contexts.
- **AI-assisted annotations:** explicit semantic review decisions supplied through ChatGPT and aggregated by the evaluator.

## Original retrieval comparison

The original comparison contains **20 questions about two synthetic PDFs**, with document/page/text references. All methods use the same embedding model and relevance dataset.

| Retrieval mode | Hit Rate@1 | Hit Rate@3 | MRR, complete ranking |
| --- | ---: | ---: | ---: |
| Semantic | 75% | 95% | 0.862500 |
| Hybrid | 65% | 80% | 0.765833 |
| German Hybrid | 60% | 95% | 0.776667 |

Hybrid combines semantic ranking and BM25 using equal-weight Reciprocal Rank Fusion with constant 60. German Hybrid applies German Snowball normalization to both document and query tokens. It restores top-3 coverage relative to ordinary Hybrid but does not outperform Semantic top-1 or MRR in this sample. No default-mode change follows from these results.

The [relevance dataset](../data/evaluation/retrieval_questions.json), [retrieval implementation](../backend/app/services/hybrid_search.py) and [comparison script](../backend/scripts/evaluate_retrieval.py) are intended public inputs. The script's default output can overwrite a historical report; use a fresh explicit output path as documented in the [README](../README.md#testing-and-evaluation).

Hit Rate@k measures at least one relevant hit. This original MRR uses the full ranking; the later experiments use **MRR@3**, which assigns zero beyond rank 3. Neither metric alone proves complete evidence for a multipart question.

## Why experimental approaches remain isolated

| Experiment | Observed result | Current decision |
| --- | --- | --- |
| Explicit document selection | 18/18 controlled routing cases, including expected rejections; no scoped leakage in the policy results | Integrated deterministic scope validation; not autonomous LLM routing |
| Automatic retrieval-mode selection | No validated rule exists; the policy preserves its fallback. All three fixed modes tie on the small local dataset of six questions per split | Disabled and not integrated into the application |
| Structural retrieval diagnostics | 32 controlled cases behaved as expected; `sufficient` is never assigned | Diagnostic only; `unknown` does not cause abstention or retry |
| Query retry | Fixed-vector fixtures improve, but the original retry rule rescues no failures with local embeddings. The fair candidate-budget baseline matches or exceeds retry complete-evidence coverage | No productive retry or automatic query rewriting |
| MMR candidate selection | On the independent synthetic test split, both Top-3 and selected MMR provide complete evidence in 5/7 evidence-bearing cases; MMR adds computation | No productive MMR selection |
| Additional LLM judges | No automated judge experiment was run and no incremental benefit was established | No extra runtime judge calls |

For MMR, lambda was selected on development data only and frozen before test evaluation. The selected value is **1.0**, giving no diversity penalty. Lower development values lost evidence. This does not demonstrate that diversification is generally ineffective; it shows no useful independent gain in the inspected experiment.

These decisions concern the current implementation and evidence. They are not general conclusions about hybrid search, retry, MMR or LLM judges.

## Synthetic answer fixtures

The offline answer-quality dataset contains **four synthetic documents, 16 cases and 32 intentionally good/faulty answer snapshots**. Initial technical source validity is 23/28 references, with 178 semantic assessment units initially pending. Those numbers test the evaluator and are **not GPT-4.1-mini performance**.

The [case dataset](../data/evaluation/answer_quality_cases.json), [synthetic snapshots](../data/evaluation/answer_quality_fixtures.json), [evaluator](../backend/evaluation/answer_quality.py) and [tests](../backend/tests/test_answer_quality.py) are intended for publication. Missing and not-applicable ratings do not count as zero or success. Claims, sources and question facets have separate denominators.

## Real answer pilot

The separate controlled pilot generated **six real model responses** from frozen contexts:

| Recorded item | Value |
| --- | ---: |
| Development / test cases | 3 / 3 |
| Evaluated factual claims | 7 |
| Evaluated question facets | 10 |
| Technically valid source references | 5/5 |
| Justified abstentions, according to annotations | 3 |
| False abstentions, according to annotations | 0 |
| Unsupported domain facts, according to annotations | 0 |
| Input tokens | 2,409 |
| Output tokens | 276 |
| Total tokens | 2,685 |
| Recorded generation calls | 6 |
| Judge calls / automatic retries | 0 / 0 |

The requested model was `gpt-4.1-mini`; recorded response metadata identifies `gpt-4.1-mini-2025-04-14`. This documents a historical execution, not guaranteed future provider availability.

### AI-assisted semantic review

The semantic review was performed through **ChatGPT as an AI-assisted reviewer**. The supplied decisions were recorded and calculated by the existing evaluator. This is **not independent human evaluation**, not an automated LLM-as-a-Judge inside DocuAgent, and not an independently established estimate of production accuracy.

| Mean on the 0–2 rubric / scored denominator | Development | Test | Aggregate |
| --- | --- | --- | --- |
| Answer correctness | 2.0 / 3 claims | 2.0 / 4 claims | 2.0 / 7 claims |
| Faithfulness | 2.0 / 3 claims | 2.0 / 4 claims | 2.0 / 7 claims |
| Citation coverage | 2.0 / 3 claims | 2.0 / 4 claims | 2.0 / 7 claims |
| Source precision | 2.0 / 2 sources | 2.0 / 3 sources | 2.0 / 5 sources |
| Completeness | 2.0 / 5 facets | 2.0 / 5 facets | 2.0 / 10 facets |

D02 contains no factual claim or source; it receives no artificial perfect factual/source rating. Its justified abstention and facet completeness are assessed separately. The three justified abstentions concern D02's telephone number, D04's shipping subquestion and T07's snack prices.

**D04 wording limitation:** the answer described shipping information as absent from the supplied “document,” although the full allowed corpus contains it outside the supplied chunk. The review accepts context-relative abstention and records that “supplied section” would be more precise. The seven-claim inventory covers domain facts; abstention statements are evaluated per facet.

**T05 schema decision:** correcting the question's false premise is recorded as `correct_answer` in the existing outcome schema, with an explicit explanation. `correct_premise` is a dataset expected behavior, not a permitted review outcome. No schema change or original-answer edit was made to force the result.

## Publication and reproducibility boundary

This public summary intentionally excludes private machine paths, response identifiers, authorization files and internal execution identifiers. Original live reports, operational journals and the complete hash-bound annotation bundle remain local and unchanged.

Public synthetic fixtures and local-model retrieval experiments can be rerun with the included code and datasets, subject to installed dependencies and a cached model. **Complete public revalidation of the real pilot's artifact chain is not offered by this summary**: the original bound live artifact is not part of the recommended publication set. Full local v2 validation requires that artifact, the derived snapshots and their matching review.

No claim is made that a future paid run will reproduce identical answers. The completed pilot must not be restarted by deleting its claim/reservation state. A live evaluation needs separate explicit authorization; no live command is included in the quick start.

## Limitations

- Small synthetic samples, with one supplied context chunk per real pilot case.
- Frozen context rather than an end-to-end retrieval test in the real pilot.
- No independent Human Evaluation or inter-rater study.
- No repeated real-model measurements or demonstrated response stability.
- Technically valid references do not imply semantically correct citations.
- No generalizable production-accuracy or multi-user security claim.
- Most dependencies and the embedding model revision are not locked for exact future reproduction.
- Local timing measurements are not load tests or cross-machine benchmarks.

The full local suite recorded **665 passing tests on 25 November 2026**. This is an automated regression result, not a measure of model accuracy.
