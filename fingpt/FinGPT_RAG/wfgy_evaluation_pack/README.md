# FinGPT RAG Failure Map Evaluation Pack

This optional pack adapts the WFGY 16-problem `ProblemMap` to common FinGPT retrieval-augmented finance workflows.

It is intended as a docs-only evaluation aid for users who run FinGPT inside RAG pipelines over filings, earnings calls, news, research notes, or lightweight time-series tools.

## What This Pack Adds

- A shared vocabulary for recurring RAG failures in financial QA.
- A compact mapping from symptom to WFGY `ProblemMap` number.
- A "look here first" guide for the FinGPT component most likely to be drifting.
- A small set of representative financial failure cases that can be turned into notebooks or regression tests.

This pack does not change FinGPT model weights, training code, or benchmark scripts.

## Scope

This pack is most useful when a FinGPT deployment includes some combination of:

- document ingestion and chunking for `10-K`, `10-Q`, earnings call, or news data
- embedding and vector retrieval for evidence selection
- prompt assembly over retrieved context
- optional tool or router steps for price / time-series lookups
- qualitative evaluation beyond simple task accuracy

## Upstream Reference

The taxonomy used here follows the WFGY `ProblemMap` project:

- WFGY ProblemMap overview: <https://github.com/onestardao/WFGY/blob/main/ProblemMap/README.md>

The stable identifiers are preserved as `No.1` through `No.16`.

## Representative Financial Failure Cases

### 1. Hallucinated risk statement in a filing answer

Example symptom:
"The company disclosed a material liquidity warning in the latest 10-K" even though the retrieved section only discusses seasonal working-capital swings.

Likely mapping:

- `No.1` hallucination & chunk drift
- `No.4` bluffing / overconfidence
- `No.8` debugging is a black box

Inspect first:

- filing parser and chunk boundaries
- top-k retrieved passages and their scores
- answer prompt requirements for citation and abstention

### 2. Broken retrieval over filings after an indexing refresh

Example symptom:
After rebuilding the index, the same question starts pulling footnotes, boilerplate risk factors, or the wrong company-year combination.

Likely mapping:

- `No.1` hallucination & chunk drift
- `No.5` semantic != embedding
- `No.8` debugging is a black box

Inspect first:

- chunk size and overlap around tables / section headers
- embedding model or vector-store configuration changes
- metadata filters for ticker, filing type, and report date

### 3. Prompt drift on multi-step financial QA

Example symptom:
The system retrieves the correct passages for revenue, operating margin, and guidance, but the final answer mixes quarterly and annual figures.

Likely mapping:

- `No.2` interpretation collapse
- `No.3` long reasoning chains
- `No.6` logic collapse & recovery

Inspect first:

- intermediate prompt instructions
- whether the answer chain separates extraction from synthesis
- whether the pipeline forces the model to reconcile units, dates, and entities before generation

### 4. Misrouted time-series or tool call

Example symptom:
A question that needs both filing evidence and recent price action is answered from text alone, or the router calls the wrong tool and ignores the retrieved filing context.

Likely mapping:

- `No.6` logic collapse & recovery
- `No.13` multi-agent chaos

This mapping is an inference for FinGPT-style tool routing because the upstream WFGY taxonomy does not define a separate tool-router-only bucket.

Inspect first:

- router prompt and tool-selection policy
- message/state handoff between retrieval and tool steps
- whether tool outputs are normalized before the final answer step

## Troubleshooting Table

| Symptom | ProblemMap | Look at first in a FinGPT pipeline |
| --- | --- | --- |
| Retrieved passages are off-topic or from the wrong company / period | `No.1`, `No.5`, `No.8` | ingestion metadata, chunking rules, embedding model, retriever filters |
| Evidence is correct but the answer misstates what the text means | `No.2`, `No.4` | answer prompt, citation policy, extraction vs. synthesis separation |
| Multi-hop questions degrade across several reasoning steps | `No.3`, `No.6` | chain design, intermediate state, decomposition prompt |
| Session follow-up loses earlier constraints or portfolio context | `No.7`, `No.9` | chat memory, conversation state persistence, context window packing |
| Tool calls fire in the wrong order or overwrite retrieved evidence | `No.6`, `No.13` | router policy, state handoff, agent role boundaries |
| Failures are hard to reproduce because traces are missing | `No.8` | retrieval logs, prompt snapshots, evaluator artifacts |
| First-call failures appear only after deploy or config refresh | `No.14`, `No.16` | startup order, environment variables, model / index version alignment |

## Full 16-Problem Mapping To FinGPT

| No. | WFGY ProblemMap category | FinGPT area to inspect first |
| --- | --- | --- |
| `1` | hallucination & chunk drift | document splitting, section boundaries, retrieval top-k |
| `2` | interpretation collapse | answer prompt, instruction template, evidence formatting |
| `3` | long reasoning chains | decomposition strategy, intermediate answer scaffolding |
| `4` | bluffing / overconfidence | abstention rule, confidence policy, evaluator prompts |
| `5` | semantic != embedding | embedding model choice, vector-store setup, metadata filters |
| `6` | logic collapse & recovery | chain control flow, retry / reset policy, router fallback |
| `7` | memory breaks across sessions | conversation state, memory store, follow-up context selection |
| `8` | debugging is a black box | retrieval traces, prompt snapshots, run-level logging |
| `9` | entropy collapse | prompt length, context packing, redundant evidence load |
| `10` | creative freeze | generation prompt only; usually low priority for factual financial QA |
| `11` | symbolic collapse | table reasoning, ticker / unit normalization, formula-like prompts |
| `12` | philosophical recursion | self-referential evaluator or critique loops |
| `13` | multi-agent chaos | multi-agent RAG, tool routing, reviewer / planner role overlap |
| `14` | bootstrap ordering | index build order, service startup dependencies |
| `15` | deployment deadlock | service orchestration across retriever / model / storage layers |
| `16` | pre-deploy collapse | secrets, model path, embedding/index version skew |

## Suggested Evaluation Flow

Use this pack as an optional stress-test layer alongside `FinGPT_Benchmark`:

1. Start with a small financial QA pipeline over one data source, such as a single `10-K` plus a simple price series.
2. Write 5 to 10 questions that expose evidence selection, temporal reasoning, citation, and tool-routing behavior.
3. For each failure, log:
   - user question
   - retrieved evidence
   - model answer
   - expected answer or accepted evidence
   - `ProblemMap` number
   - first FinGPT component inspected
4. Fix one failure mode at a time and rerun the same questions after each change.

## Minimal Starter Set

If you want a compact first notebook or docs demo, start with these three cases:

- Retrieval drift across two similar filings for the same ticker
- Hallucinated risk disclosure from a correctly retrieved filing chunk
- Filing-plus-price question where the tool router ignores the price lookup

That set already covers retrieval, reasoning, and routing failures without requiring changes to FinGPT core code.

## Related FinGPT Modules

- `fingpt/FinGPT_RAG`
- `fingpt/FinGPT_Benchmark`
- `fingpt/FinGPT_MultiAgentsRAG`

## Attribution

This evaluation pack borrows the failure taxonomy from the WFGY `ProblemMap` and adapts it to FinGPT usage patterns. Users who want the original definitions and companion materials should consult the upstream WFGY repository.
