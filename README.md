# Guaranteed Structured Generation: A Grammar-Constrained Decoding Engine for Large Language Models

Nov 23, 2026 · @Rahul

**Abstract.** This project builds a decoding engine that guarantees a language model's output conforms to a formal specification: a regular expression, a JSON Schema or a context-free grammar. At every generation step, the engine masks every token that cannot lead to a valid output, so invalid output becomes impossible rather than unlikely. The engine is written from the sampling loop upward on a small open model (Qwen2.5-0.5B). It compiles grammars into finite-state and pushdown automata, precomputes token masks over a vocabulary trie, and skips model calls for forced text using jump-forward decoding. It is evaluated against XGrammar, llguidance, Outlines and llama.cpp on validity rate, per-token mask overhead and grammar compilation time.

**Project 1 of 6** in the inference-systems series. Development runs on a CPU-only Ubuntu workstation (16 GB RAM); no GPU is required for this project.

## 1. Problem and motivation

LLM agents act on the world through structured output: tool calls, function arguments, API payloads, database queries. One malformed token, such as a missing brace or an unquoted key, breaks the downstream program. Instructions in the prompt make valid output likely but never certain.

The gap is large. When OpenAI launched Structured Outputs in August 2024, it reported that `gpt-4-0613` scored under 40% on its complex-schema evaluation through prompting alone, while constrained decoding reached 100% ([OpenAI](https://openai.com/index/introducing-structured-outputs-in-the-api/)). Constrained decoding is now standard infrastructure. vLLM ships XGrammar and llguidance as its structured-output backends ([vLLM docs](https://docs.vllm.ai/en/latest/features/structured_outputs/)), and XGrammar-2 (2026) extends the idea to multi-tool agent formats ([arXiv 2601.04426](https://arxiv.org/abs/2601.04426)).

The engineering problem is speed. A modern vocabulary holds 128,000–150,000 tokens, and the set of legal tokens must be recomputed at every step. If that takes milliseconds, it becomes the bottleneck of generation. The reference point: llguidance computes a mask in about 50 µs per token on a 128k vocabulary ([llguidance](https://github.com/guidance-ai/llguidance)).

## 2. Background: the concepts this project rests on

### 2.1 How a model chooses a token

At each step the model outputs a **logit** zᵢ for every token i in its vocabulary of size V. Sampling turns logits into a choice. With temperature T, the probability of token i is:

```latex
p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{V} \exp(z_j / T)}
```

**Greedy** decoding takes the highest logit. **Top-k** keeps only the k best tokens; **top-p** (nucleus) keeps the smallest set whose probabilities sum to p. The chosen token is appended, and the model's **KV cache** stores past computation so the next step only processes one new token.

### 2.2 Constraining by masking

To forbid token i, set zᵢ = −∞ before sampling. Its probability becomes zero, and the allowed tokens keep their relative probabilities. The whole project reduces to one question asked at every step: *which tokens are allowed right now?*

### 2.3 Formal languages and automata

- A **regular expression** describes a regular language and compiles to a **finite-state machine** (NFA, then DFA). A date pattern like `\d{4}-\d{2}-\d{2}` is regular.
- **JSON** allows unlimited nesting (objects inside arrays inside objects), so it is **context-free**. Recognising it needs a **pushdown automaton**: a state machine with a stack that remembers which brackets are open.
- A **context-free grammar (CFG)** written in **EBNF** describes any such language. An **Earley parser** can recognise any CFG incrementally, one character at a time.
- **Python** is not context-free because indentation carries meaning, so it needs a lexer that tracks indentation state, or a restricted subset.

### 2.4 The token–grammar mismatch

Grammars are defined over characters; models emit **BPE tokens**, which are byte strings of varying length. A single token such as `"},` closes a string, closes an object and starts a new field, crossing three grammar symbols. A token is legal only if the automaton can consume *all* its bytes from the current state. Checking 150,000 tokens byte by byte at every step is too slow, which is why the techniques below exist.

### 2.5 Making masks fast

- **Precomputed index (Outlines, 2023).** For a finite-state machine, compute once which tokens are legal in each state; at runtime the mask is a table lookup ([arXiv 2307.09702](https://arxiv.org/abs/2307.09702)).
- **Context-independent tokens (XGrammar, 2024).** Most tokens' legality depends only on the current grammar position, not on the stack beneath it, so they can be precomputed; only the remaining few are checked at runtime. XGrammar reports up to 100× speedups ([arXiv 2411.15100](https://arxiv.org/abs/2411.15100)).
- **Vocabulary trie (llguidance).** Store all tokens in a byte trie; walking the trie with the parser prunes every token sharing a rejected prefix at once.
- **Jump-forward decoding.** When the grammar allows only one continuation (e.g. `"name": "` after `{`), append it directly without running the model.
- **Token healing.** The last token of a prompt can split a word unnaturally; backing it up and re-generating with a constraint avoids distorted output.

## 3. System design: components of the engine

The engine is a pipeline of six components, each a separate module with its own tests.

1. **Grammar frontend.** Accepts three input forms: a regular expression, a JSON Schema (generated from a Pydantic model with `model_json_schema()`), or an EBNF grammar. All three lower to one internal grammar representation.
2. **Grammar compiler.** Regular grammars compile to a DFA (Thompson construction, then subset construction). JSON Schema and EBNF compile to a pushdown automaton, or are handed to an Earley recogniser.
3. **Vocabulary preprocessor.** Decodes every token ID to its exact byte string (handling byte-fallback and leading-space markers) and builds a byte trie of the whole vocabulary. Runs once per tokenizer and is cached to disk.
4. **Mask generator.** Given the automaton's current state, walks the trie and emits a bitset of allowed token IDs. Masks for states without stack dependence are cached and reused.
5. **Sampler integration.** Two front doors: a hand-written decode loop with full control over the KV cache, and a Hugging Face `LogitsProcessor` so the engine works inside `model.generate`.
6. **Jump-forward controller.** Detects when the automaton has exactly one legal continuation, appends those bytes, re-tokenises the boundary, and resumes sampling.

## 4. Implementation plan: milestones, deliverables, acceptance criteria

| # | Milestone | Deliverable | Done when |
| --- | --- | --- | --- |
| M1 | Reference decode loop | `generate()` with KV-cache reuse, greedy, temperature, top-k, top-p | Greedy output is token-identical to `model.generate` on 20 prompts |
| M2 | Regex constraint | Regex → DFA compiler; per-step masking | 1,000 samples match the regex 100%; mask time recorded |
| M3 | Vocabulary trie and mask cache | Byte trie, bitset masks, per-state cache | Mask latency reported before and after; same outputs as M2 |
| M4 | Flat JSON Schema | Schema → grammar for objects, strings, numbers, enums, booleans | 100% of outputs pass Pydantic `model_validate_json` |
| M5 | Nested grammars | Pushdown automaton or Earley recogniser; EBNF input | Arrays of objects, nested objects and recursive schemas validate 100% |
| M6 | Jump-forward decoding | Forced-continuation detection and boundary re-tokenisation | Model calls saved reported per schema; outputs still valid |
| M7 | Benchmark and report | Harness, plots, README, write-up | Section 5 experiments complete and reproducible with one command |

Model: Qwen2.5-0.5B-Instruct as the main model; GPT-2 (50k vocabulary) as a second tokenizer to study vocabulary-size effects. Both run comfortably in 16 GB RAM. Expected effort: three to four weeks part-time.

## 5. Evaluation: baselines, metrics, experiments

**Baselines.** Unconstrained generation; prompt-only ("respond in JSON matching this schema"); XGrammar; llguidance; Outlines; llama.cpp GBNF grammars ([llama.cpp grammars](https://raw.githubusercontent.com/ggml-org/llama.cpp/HEAD/grammars/README.md)).

**Test suite.** 30–50 JSON Schemas in four tiers: flat objects; enums and numeric ranges; nested objects and arrays; recursive definitions. Plus 5–10 regex patterns and 3 EBNF grammars (arithmetic, a SQL subset, a Python subset).

| Metric | Definition | Why it matters |
| --- | --- | --- |
| Validity rate | Share of outputs that parse and validate | The guarantee itself |
| Mask latency | µs per token, p50 and p99 | The cost of the guarantee |
| Compile time | ms to turn a schema into an automaton | First-request latency |
| Model calls saved | % of tokens produced by jump-forward | Speedup beyond masking |
| End-to-end throughput | tokens/s, constrained vs unconstrained | Total overhead |
| Answer quality | Accuracy on a small extraction task, constrained vs prompt-only | Whether the constraint distorts content |

**Experiments.**

1. Validity across all baselines and schema tiers.
2. Mask latency vs vocabulary size (GPT-2 at 50k vs Qwen at 151k).
3. Mask latency vs grammar complexity (flat vs deeply nested).
4. Ablation: no cache, cache only, cache plus trie, plus jump-forward.

Report mask overhead separately from the model's forward pass. On a CPU the forward pass dominates wall-clock time, so end-to-end numbers alone would hide the engine's own cost. Expect the Rust and C++ libraries to be faster than a Python engine; the write-up should explain exactly where the gap comes from.

## 6. What you will learn and what it demonstrates

**What you will learn:**

- **Tokenizer internals.** How BPE merges build a vocabulary, how byte-level tokens and leading-space markers work, and why a token is not a word or a character.
- **The decode loop.** Logits, softmax, temperature, top-k and top-p sampling, and how the KV cache makes each step process a single new token.
- **Automata theory in practice.** Regex to NFA to DFA, pushdown automata for nesting, and incremental Earley parsing.
- **Performance data structures.** Tries, bitsets and memoisation, chosen to turn a 150,000-item check into a microsecond lookup.
- **Measurement discipline.** Microbenchmarks, p50/p99 percentiles, and separating one component's cost from the whole system's.
- **The production landscape.** How vLLM and SGLang plug structured output into serving, and what XGrammar and llguidance do differently.

**What it demonstrates to a reviewer:**

- You can change the core generation loop of an LLM, not only call an API.
- You build guarantees by construction rather than by prompting.
- You can optimise to microsecond budgets and prove it with measurements.
- You understand the reliability layer that agent and tool-calling platforms depend on.

## 7. Extensions

- **Structural tags.** Let the model write free text and switch into a grammar only inside a `<tool_call>` block, as XGrammar-2 does for agents.
- **Constrained speculative decoding.** A small draft model proposes several tokens, the main model verifies them in one pass, and the grammar mask applies to both. Measure acceptance rate with and without the constraint.
- **Python subset.** Constrain to a restricted Python grammar with indentation tracking, and confirm every output with `ast.parse`.
- **Batched masks.** Compute masks for a batch of requests with different schemas in parallel, and measure how overhead scales with batch size.
- **Upstream contribution.** File an issue or PR to XGrammar, llguidance or vLLM's structured-output code based on a gap your benchmarks reveal.

## 8. Glossary and references

| Term | Meaning |
| --- | --- |
| Logits | Raw score for every vocabulary token before sampling |
| Greedy, temperature, top-k, top-p | Rules for choosing the next token from the scores |
| KV cache | Stored attention keys and values so past tokens aren't recomputed |
| BPE, vocabulary | Byte-pair encoding tokenizer; its full list of tokens |
| Constrained / guided / structured decoding | Restricting which tokens may be generated |
| Logit masking | Setting forbidden tokens' logits to −∞ |
| LogitsProcessor | Hugging Face hook that edits logits before sampling |
| Regular language, FSM, DFA | Patterns without nesting; the state machine that recognises them |
| Context-free grammar (CFG) | Rules that allow unlimited nesting |
| Pushdown automaton (PDA) | State machine with a stack, for context-free languages |
| Earley parser | Incremental parser for any context-free grammar |
| EBNF, GBNF | Grammar notations; GBNF is llama.cpp's dialect |
| JSON Schema, Pydantic | Standards for describing the required shape of JSON |
| Token–grammar mismatch | Tokens crossing grammar-symbol boundaries |
| Token healing | Re-generating the prompt's last token under the constraint |
| Token mask, bitset | One bit per token marking it legal or not |
| Context-independent tokens | Tokens whose legality can be precomputed per grammar position |
| Vocabulary trie | Prefix tree of all token byte strings |
| Jump-forward decoding | Emitting forced text without a model call |
| Structural tags | Grammars activated only within tagged regions of output |
| Function / tool calling | Model output that a program executes |
| Speculative decoding | Drafting several tokens cheaply and verifying them in one pass |

**References**

1. Willard and Louf, *Efficient Guided Generation for Large Language Models* (Outlines), 2023. [arXiv 2307.09702](https://arxiv.org/abs/2307.09702)
2. Dong et al., *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models*, 2024. [arXiv 2411.15100](https://arxiv.org/abs/2411.15100)
3. XGrammar-2, 2026. [arXiv 2601.04426](https://arxiv.org/abs/2601.04426); [MLC blog](https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation)
4. llguidance. [GitHub](https://github.com/guidance-ai/llguidance)
5. OpenAI, *Introducing Structured Outputs in the API*, Aug 2024. [openai.com](https://openai.com/index/introducing-structured-outputs-in-the-api/)
6. vLLM structured outputs. [docs.vllm.ai](https://docs.vllm.ai/en/latest/features/structured_outputs/)
7. llama.cpp GBNF grammars. [README](https://raw.githubusercontent.com/ggml-org/llama.cpp/HEAD/grammars/README.md)
