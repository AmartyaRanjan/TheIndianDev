# Implementation Plan — Adversarial Self-Play for Code Bug Detection

Source spec: `2rlproj.pdf` (method §2, ablations §3, evaluation §4). This file turns it into a build order.

## Status (2026-10-06): full details and the GPU runbook are in `PROGRESS.md`

| Phase | State |
|---|---|
| 0 | uv + Python 3.12 venv (torch 2.13/cu130, vLLM 0.31, transformers 5.17, PEFT 0.21); Qwen2.5-Coder-1.5B-Instruct downloaded. GPU check passed in the ~6 GB left free by other services (vLLM at 24%, eager mode, 4096-token context) |
| 3 (measured) | Base 1.5B on real-dev: zero-shot accuracy 0.370 / line_exact 0.019 / FPR 0.268; few-shot 0.013 / 0.157 / 0.999. It strongly prefers "line 10". Detector format failures 0.1% (no detector warmup needed). **Injector: 0/256 gate-passing mutants**, so a format warmup was added (`scripts/warmup_injector.py`: SFT on 2,628 gated AST-engine examples, loss 0.59→0.11). After warmup: gate pass 48%, format failures 3%, 32/32 functions with ≥1 valid mutant. Every arm starts from `runs/warmup_injector/injector` (`injector.init_adapter`) |
| 3 (Gemini) | AI Studio key in `.env` (git-ignored). 30-example dev samples: 3.1 Pro default 0.667 accuracy at ~5.5k thinking tokens/request; Pro `low` 0.800 at ~470; 3.8 Flash 0.833 at ~1.6k. Full dev run awaits a model/cost choice |
| GPU smoke | 0.5B: the loop runs end to end on the GPU (13–20 s/step at tiny scale), but the 0.5B model can't follow the output format (100% format failures at step 0), so it is unusable even as a smoke-scale policy |
| 1 | Done. Taxonomy v1 (with the §4.1 deviations), libcst mutators, AST-diff classifier; every class round-trips |
| 2 | Done. Corpus: 24,048 train / 952 val KodCode functions, tests pass twice. Held-out: 3,500 real bug pairs (693 dev / 2,807 test) from PyPIBugs + TSSB-3M, every class ≥143, 0 near-duplicates of the corpus. Synthetic-val: 836 gated mutants (`none_guard_omission` only 21: KodCode rarely guards against None) |
| 3 | Prompts/parsers, metrics, linter baseline (dev: line_exact 0.091, FPR 0.857), Claude baseline code (not run: needs a key and approval to spend ~$13–46 on dev), `scripts/eval.py`. **Waiting on GPU:** base-model zero/few-shot + format-compliance check |
| 4–5 | Code done and tested on CPU (tiny random Qwen2 + scripted generator): GRPO, policy/adapters, vLLM + HF generators, gate G1–G6, rewards, pools, curriculum, checkpoint/resume, `scripts/train.py`, `scripts/run_queue.py`. **Waiting on GPU:** first real runs and the TRL comparison |
| 6+ | Not started (needs GPU). Not written yet: A4 PPO critic, A5 arm files, SFT baseline |

## 0. Decisions locked in

| Item | Decision |
|---|---|
| Compute | Local NVIDIA L4 (24 GB), after the other GPU services are stopped |
| Base model | Qwen2.5-Coder-1.5B-Instruct for both players: one frozen base, two LoRA adapters (injector, detector) |
| Language | Python |
| Scope | Tier 1 first (main + A1, A2, A3, 3 seeds), then Tier 2 (A4, A5), then Tier 3 (A6, A7) |
| ChatGPT / Claude | Baseline only (zero/few-shot). GRPO needs gradients through the weights; closed APIs don't expose them |

## 1. Positioning (read before building)

Reviewers will compare against three prior works:

- **BugLab** (Allamanis et al., NeurIPS 2021). Co-trains a bug *selector* and a *detector* on Python and beats random-mutation training on **PyPIBugs** (2,374 real bugs). That is essentially our A1 result, with GNN/transformer models instead of LLMs. So for us A1 is a *replication with LLMs + GRPO*. The novelty has to come from A2–A5 and the dynamics curves (PDF §3.2). Use PyPIBugs as a held-out set so the numbers are directly comparable.
- **Self-play SWE-RL** (Wei et al., ICML 2026, arXiv 2512.18552). Trains one LLM in two roles (inject bugs, repair bugs) with RL at repo level. It shows LLM bug-injection self-play works but does not ablate the failure modes. We differ in three ways: two separate policies, function-level localization + classification, and the controlled ablations.
- **TSSB-3M** (Richter & Wehrheim, MSR 2022) and "Can we learn from developer mistakes?" (2022). They show mutation-trained detectors transfer poorly to real bugs. Cite them as the motivation for the sim-to-real gap.

Two notes on the PDF annotations:
- *"github commits"*: GitHub bug-fix commits form the **held-out** set (PDF §2.4) and never enter training. Training mutations come only from the injector, or from the AST engine in A1.
- *"already a dataset on kaggle"*: if that dataset holds real bugs, it is a candidate for the held-out set. If it holds synthetic mutations, it is a candidate static baseline for A1. Need the link to decide.

## 2. Environment and hardware budget

**Prerequisites**
1. **Free the L4.** About 21 GB is in use by ollama `llama-server`, fish-speech, two indicf5 processes, four `/opt/venv` services, and another user's `simple-s2s` (`/home/nupriya/...`). Agree with whoever owns them before stopping them. Runs take days, so the GPU has to stay free.
2. **Python version.** The system Python is 3.14, which vLLM and PyTorch wheels may not support. Install `uv` and pin a Python 3.12 venv.
3. **Disk.** About 66 GB is free. Keep LoRA pool snapshots only while a run is live; at the end keep the final adapter plus one every 100 steps. Download only the TSSB-3M subset you need.

**Memory plan on about 22 GB (one process, generation and training on the same GPU)**

| Component | Approx. |
|---|---|
| vLLM engine: frozen base (bf16) + KV cache, multi-LoRA on, prefix caching on | ~8 GB (`gpu_memory_utilization≈0.35`) |
| Training copy of base (bf16, frozen) + LoRA adapters + AdamW state | ~4 GB |
| Activations (gradient checkpointing, micro-batch 4–8) | ~3–5 GB |
| Headroom | ~4 GB |

**Key trick.** The base stays frozen and only the LoRA adapters train, so vLLM never needs weight syncing. After each update, save the adapter (~70 MB at r=32) and generate with a new `LoRARequest`. This has two side benefits:
- The KL reference model costs nothing: it is the same base with adapters disabled (`with model.disable_adapter()`).
- Checkpoint pools are just old adapter files.

Compute log-probs only on completion tokens, in chunks. Otherwise the 151k-token vocabulary makes the full logits tensor the biggest memory cost.

## 3. Repository layout

```
rl/
  pyproject.toml            # uv-pinned: torch, vllm, transformers, peft, libcst, datasets, wandb, pytest
  configs/
    base.yaml               # defaults (§5)
    arms/                   # one override file per ablation arm: full.yaml, a1_frozen.yaml, a1_uniform.yaml, ...
  src/advbug/
    taxonomy.py             # TAXONOMY_VERSION, class defs, prompt descriptions
    mutation/
      operators.py          # libcst mutators, one per class (keeps formatting)
      classify.py           # AST diff -> class, priority-ordered; used by the gate AND held-out labelling
    data/
      schema.py             # Example(code, bug_lines, bug_class, source, split, tests)
      corpus.py             # clean training corpus (KodCode): filter, dedupe, verify tests
      heldout.py            # PyPIBugs + TSSB-3M (+ BugsInPy later) -> unified Examples
    sandbox.py              # run a snippet's tests in a worker process: timeout, rlimits, no network
    game/
      prompts.py            # injector/detector prompts + strict output parsers
      gate.py               # validity gate G1–G6 (§4.4)
      rewards.py            # detector binary/partial; injector fool-rate + diversity + subtlety
      pool.py               # adapter checkpoint pools for both players
      curriculum.py         # A7
    rl/
      policy.py             # PEFT model with named adapters (injector, detector, critics)
      rollout.py            # vLLM multi-LoRA generation
      grpo.py               # group-normalized advantages, loss, KL
      ppo.py                # value head + GAE (A4)
      loop.py               # one co-training step (§4.5), checkpoint/resume
    eval/
      metrics.py            # exact-line, within-N, per-class F1, false-positive rate on clean code, ECE
      dynamics.py           # entropy, fool rate, sim-to-real gap, advantage variance, zero-variance groups
      baselines.py          # linters, base-model zero/few-shot, Claude zero/few-shot, SFT on static mutations
      stats.py              # paired bootstrap, McNemar, Holm, seed aggregation
  scripts/                  # build_corpus.py, build_heldout.py, train.py, eval.py, run_queue.py, figures.py
  tests/
```

Write metrics to local JSONL on every run. W&B (offline mode is fine) is for viewing; the figures are built from the JSONL.

## 4. Components

### 4.1 Taxonomy v1 (fixed before any training run)

Classes must be **disjoint**, otherwise per-class F1 is meaningless. The AST-diff classifier checks them in this order and the first match wins:

| # | Class | AST signature (original → mutant) |
|---|---|---|
| 1 | `negation` | `not` added/removed; `==`↔`!=`, `is`↔`is not`, `in`↔`not in` |
| 2 | `boundary_shift` (off-by-one) | `<`↔`<=`, `>`↔`>=`; integer literal ±1; `+1`/`-1` added or removed |
| 3 | `comparison_op` | any other comparison operator change |
| 4 | `boolean_op` | `and`↔`or` |
| 5 | `arithmetic_op` | BinOp / AugAssign operator change |
| 6 | `argument_swap` | two positional arguments of one call swapped |
| 7 | `none_guard_omission` | an `if x is None: ...` guard (body ≤2 lines) removed; an `x is not None and` conjunct removed; `x or default` → `x` |
| 8 | `type_conversion` | builtin conversion (`int/str/float/list/dict/set/tuple/bool`) added, removed or changed |
| 9 | `wrong_return` | `return` expression changed or replaced by `None` (when no class above matches) |
| 10 | `variable_misuse` | one variable read replaced by another variable in scope |
| — | `no_bug` | detector-only label |

**Two deviations from the PDF list (please confirm):**
- *Incorrect loop bound* is not its own class, because it overlaps off-by-one and variable misuse. It becomes a secondary **location tag** for analysis, alongside `condition`, `return` and `call_arg`.
- *Resource/handle leak* moves to v2. Unit tests cannot tell a leak from correct code, so the non-equivalence gate (G6) cannot verify it, and the test-backed corpus rarely opens files.

### 4.2 AST mutation engine (`mutation/`)

- **`operators.py`.** For each class, list every applicable site in a function. `mutate(code, cls, rng)` applies one site via **libcst**, which preserves formatting and comments. Don't use `ast.unparse`: it reformats the whole function, and the detector could learn to spot reformatted code.
- **`classify.py`.** Given `(original, mutant)`, return the class or `None`. It diffs the two `ast` trees down to the smallest differing subtree, then tests the class signatures in priority order.
- **Three uses:** generating the A1-uniform baseline data, the gate's label check (G5), and labelling real bugs as `classify(fixed, buggy)`.

### 4.3 Data

**Clean training corpus: KodCode** (447k Python problem + solution + pytest triplets)
- Keep solutions of 8–40 lines whose tests pass in the sandbox within 2 s.
- Dedupe with exact match plus MinHash.
- Target ~20k training functions and ~1k synthetic-validation functions.
- The tests are what make the non-equivalence check (G6) possible. That is why this corpus beats untested GitHub code here.

**Held-out real bugs** (never used in the training loop)
- **Sources:**
  - **PyPIBugs**: 2,374 bugs. The metadata jsonl comes from Microsoft; fetch each file at its commit from GitHub. Directly comparable to BugLab.
  - **TSSB-3M**: sample ~3k bugs from SStuB patterns that map to the taxonomy.
  - **BugsInPy**: 493 bugs, many with multiple changed hunks. Tier 2 only, localization only.
- **Labelling each bug:**
  - Extract the enclosing function: the pre-fix version is the buggy example, the post-fix version is a paired clean negative.
  - Bug lines are the lines changed by the fix, in the buggy version's numbering.
  - Class is `classify(fixed, buggy)`. If that fails, the class is `other`, which is scored for localization only.
  - Drop functions longer than 80 lines.
- **Splits:** `real-dev` (20%) is for training curves and sanity checks. `real-test` (80%) is scored once per final checkpoint and nothing else. Commit the split IDs and a hash.
- **Leakage controls:**
  - Near-duplicate check between the training corpus and the held-out set.
  - Exclude held-out repos from any non-KodCode code you add later.
  - The base model's pretraining contamination can't be controlled. It affects every arm equally; say so in the paper.

### 4.4 The game: prompts, gate, rewards

Code is shown with line-number prefixes (` 7| ...`).

**Injector output** (strict, parsed):
```
<type>boundary_shift</type>
<lines>7-7</lines>
<new>    for i in range(len(xs) - 1):</new>
```
- The edit replaces lines a..b (b−a ≤ 2) with 0–3 lines. An empty `<new>` is a deletion, used for `none_guard_omission`.
- The detector's target line is the first changed line in the mutant. For a deletion, it is the line that now sits at position a.

**Detector output:** `<line>7</line><type>boundary_shift</type>` or `<type>no_bug</type>`. No chain-of-thought by default, for cost; it is available as a config flag.

**Validity gate.** Each check is logged separately. A3 turns off G2–G6.
- **G1** The output parses and the line range is valid.
- **G2** The edit is minimal: at most 3 lines and a small token edit distance.
- **G3** The mutant parses and compiles.
- **G4** No shortcut artifacts: no added or changed comments or docstrings, indentation preserved, not a whitespace-only change.
- **G5** `classify(original, mutant) == claimed type`.
- **G6** Non-equivalence: the original passes its tests (precomputed) and the mutant fails at least one; a timeout counts as a failure. G6 runs only after G5 passes, so arbitrary code written by the injector never executes.

**G6 is the most important check.** Without it, the injector learns *equivalent mutants* that no detector can ever catch, e.g. `a+b` → `b+a` labelled as a bug. That is pure reward hacking. Run tests with a lightweight per-worker runner rather than one `pytest` process per mutant; pytest startup would dominate the step time.

**Rewards (per sample)**
- **Detector, binary (default):** 1 if the input is a mutant and both the exact line and the type match, or if the input is clean and the answer is `no_bug`. Otherwise 0.
- **Detector, partial (A6):** `0.5·line_score + 0.5·type_match`, where `line_score` is 1 / 0.5 / 0.25 for a line error |Δ| of 0 / 1 / ≤3.
- **Injector:** −1 if the gate fails. Otherwise `fool_rate + λ_div·div + λ_sub·sub`:
  - `fool_rate` = 1 − the detector's mean correctness over its K_D samples on this mutant, a value in [0, 1].
  - `div` = −log p̂(class), normalized to [0, 1], where p̂ is a moving-average frequency of the class in the injector's recent valid outputs. This is part of the A2 mechanism.
  - `sub` = 1 − normalized token edit distance, so smaller edits score higher.
  - With the gate off (A3), invalid samples are not penalized; scoring uses the injector's *claimed* line and type.

### 4.5 One co-training step (generation shared between players)

1. **Generate mutants.** Sample B clean functions. The current injector produces K mutants per function, and all B·K candidates go through the gate.
2. **Detector pass.** The current detector produces K_D samples for every valid mutant and for a batch of clean functions.
3. **Shared use of that pass.** The same detector samples give each mutant's `fool_rate` (the injector's reward) *and* form the detector's GRPO groups. The detector trains on a subsample: 64 mutants plus 16 clean functions.
4. **Pools (when on).** 25% of the detector's mutant inputs come from a random past injector. Injector mutants are also scored against a random past detector (greedy, one sample); `fool_rate` blends current and past as 0.7 / 0.3.
5. **Update.** GRPO updates both players at once. Injector groups are the K mutants of one function; detector groups are the K_D answers to one input. Snapshot both adapters into the pools every P steps.

Static arms (A1) skip injector training. Mutants come from the frozen step-0 injector (A1-frozen) or from `operators.mutate` with a uniformly sampled class (A1-uniform). Detector batch size and step count stay identical to the main arm.

### 4.6 RL core

**Why a custom loop, not TRL `GRPOTrainer`.** Two players share a frozen base, their rewards depend on each other, they draw from adapter pools, and A4 swaps the algorithm. None of that fits a single-policy trainer. The custom loop is validated against TRL on the single-player case (Phase 4).

**GRPO**
- Advantage: `A_i = (r_i − mean_g) / (std_g + 1e-4)`.
- Token-level policy-gradient loss.
- KL to the reference model (adapter disabled) with β=0.01. Setting β=0 saves ~20% compute.
- μ=1 update per batch. Training is fully on-policy, so the clip never activates and no old-log-prob pass is needed.

**PPO (A4)**
- The critic is a separate LoRA adapter plus a scalar value head on the same base.
- GAE with γ=1, λ=0.95.
- Same rollouts per step and the same μ as GRPO, so the advantage estimator is the only difference. That is exactly the A4 question.

**Logged every step:** reward mean/std, advantage variance, the **fraction of zero-variance groups** (all samples got the same reward, so no gradient; central to A5), KL, format-failure rate, and gate pass rate per check.

### 4.7 Curriculum (A7): definition to confirm

The PDF wording supports more than one reading. Proposed:
- **Phase A** (detector's rolling synthetic accuracy < 0.5): cap the injector's fool-rate term at 0.5, so there is no extra reward for bugs a weak detector can never catch.
- **Phase B** (accuracy ≥ 0.5): enforce a minimum subtlety. A mutant caught by all K_D detector samples is treated as a gate failure: injector reward −1, and it is dropped from the detector batch.

### 4.8 Evaluation

- **Decoding.** Greedy decoding for accuracy. For confidence, take 8 samples at T=1 and use the vote share; this feeds ECE and reliability diagrams.
- **Metrics:**
  - exact-line accuracy; within-1 and within-3 line accuracy
  - per-class F1 and macro-F1 (on bugs that map to the taxonomy)
  - false-positive rate on clean (post-fix) functions
- **Baselines:**
  1. ruff/pyflakes/pylint warnings mapped to lines (localization only).
  2. Base Qwen2.5-Coder-1.5B zero/few-shot (= step 0).
  3. Claude zero/few-shot via API, with the same prompt and parser.
  4. **An SFT detector trained on the static mutation set**, because reviewers will ask "why RL at all?"
  5. A1-uniform (GRPO on static mutations).
- **Dynamics (PDF §3.2)**
  - Every step, from the rollouts: injector class entropy, fool rate (win rate), gate pass rate.
  - Every 25 steps: real-dev accuracy, and from it the **sim-to-real gap** = rolling on-policy synthetic accuracy − real-dev accuracy.
  - Also track accuracy on the fixed uniform-AST synthetic-val set, a yardstick that does not move during training.
- **Statistics:**
  - 3 seeds per arm, reported as mean ± std.
  - Per-example paired tests of each arm against the main arm: McNemar on exact match, paired bootstrap with 10k resamples, Holm correction across arms.
  - Hierarchical bootstrap over seeds and examples for confidence intervals.
- **Qualitative audit** (A3 and the main arm): 50 random mutants per arm, rated blind for realistic / single-cause / correct label.

## 5. Default hyperparameters (adjust during the Phase 6 pilot, then freeze)

| Param | Value |
|---|---|
| LoRA | r=32, α=64, all linear layers |
| Optimizer | AdamW, lr 1e-5, 10 warmup steps, then constant |
| Injector | B=16 functions × K=8 → 128 mutants/step; T=1.0; ≤128 new tokens |
| Detector | K_D=4; scores all valid mutants; trains on 64 mutants + 16 clean; ≤32 new tokens |
| KL β | 0.01 |
| Steps per run | 400 (re-set after the pilot) |
| λ_div / λ_sub | 0.1 / 0.05 |
| Pools | snapshot every 20 steps, keep last 10; 25% pooled detector data; 0.3 pooled-detector weight |
| Eval cadence | real-dev every 25 steps |
| A5 sweep | K ∈ {2,4,8,16} with B·K = 128 held constant, so rollouts per step don't change |

## 6. Run matrix and L4 budget

**Estimated step time** for an adversarial arm on the L4, to be measured in Phase 6:
- generation ~20–30 s
- sandbox ~5–10 s
- training both adapters ~60–120 s

That comes to **~1.5–2.5 min/step**, or 10–17 h per 400-step run. Static arms take ~7–11 h. Only one run fits on the GPU at a time, so runs go through a queue and resume from checkpoints.

| Tier | Arms | Runs (×3 seeds) | ≈ GPU time |
|---|---|---|---|
| 1 | full; A1-frozen; A1-uniform; A2-off (no diversity bonus, no pools); A3-off (no gate) | 15 | ~7 days |
| 2 | A4-PPO; A5 K∈{2,4,16} adversarial; A5 K∈{2,4,8,16} on A1-uniform (the stationary-opponent control A5's question requires) | 24 | ~12 days |
| 3 | A6-partial credit; A7-curriculum | 6 | ~3 days |

The SFT baseline, zero-shot baselines and the Claude API evaluation are cheap by comparison.

If the Phase 6 pilot shows Tier 1 won't fit, choose **before** Tier 1 starts: set β=0, cut steps to 300, or run the A5 stationary control with 2 seeds.

## 7. Phased roadmap

| Phase | Build | Exit check |
|---|---|---|
| 0 | Free the GPU; uv + Python 3.12; dependencies; load Qwen-1.5B in vLLM with LoRA enabled | vLLM generates with a dummy adapter; generation and training fit in memory together |
| 1 | `taxonomy.py`, `mutation/operators.py`, `mutation/classify.py` + tests | `classify(f, mutate(f, c)) == c` for ≥99% of 1k functions, for every class; formatting preserved |
| 2 | `data/corpus.py`, `sandbox.py`, `data/heldout.py` | ~20k clean functions with passing tests; held-out class distribution report; split hashes committed |
| 3 | Prompts + parsers, `eval/metrics.py`, baselines (base zero-shot, Claude, linters) | Baseline table on real-dev + synthetic-val. Base-model format compliance ≥80%; otherwise add one short format-SFT warmup shared by **every** arm |
| 4 | `rl/` GRPO; single-player detector on A1-uniform | Synthetic accuracy rises; the curve matches TRL `GRPOTrainer` on the same setup, within noise |
| 5 | Injector, gate, rewards, pools, full loop, dynamics logging | One full-config run is stable for 400 steps; audit of 50 mutants shows no equivalent or artifact mutants getting past the gate |
| 6 | Pilot: 1 seed × Tier 1 arms × 100 steps | Step time measured → final step count; hyperparameters frozen; arm configs committed (pre-registered) |
| 7 | Tier 1 × 3 seeds → analysis → Tier 2 → Tier 3 | Each tier analysed before the next starts |
| 8 | Figures (entropy, win rate, sim-to-real gap per arm), statistics, Dockerfile, release of code + splits | Results reproduce from released configs |

## 8. Verification

- **Unit tests (`pytest`):**
  - mutator/classifier round-trip for every class
  - the gate rejects crafted bad cases: comment injection, `a+b`→`b+a`, whitespace-only edit, label mismatch, non-parsing code
  - parsers handle malformed outputs
  - reward functions match hand-computed cases
  - GRPO advantages and loss match a hand-computed toy batch
- **Smoke run.** Qwen2.5-Coder-0.5B for 10 steps through the full loop, including checkpoint and resume.
- **RL correctness:**
  - Phase 4 TRL comparison.
  - Training the injector against a frozen detector must raise the fool rate.
  - The Phase 5 audit confirms the gain comes from real bugs, not gate loopholes.
- **Evaluation consistency.** The training-time evaluator and `scripts/eval.py` give the same real-dev score for the base model.

## 9. Open items

- Link to the Kaggle dataset (§1).
- Confirm the taxonomy deviations (§4.1) and the A7 definition (§4.7).
- Claude API key and budget for the LLM baseline. It is small: about 5k short prompts per prompt variant.
- 5 seeds for Tier 1 if time allows. RL variance is high; 3 is the minimum the PDF commits to.
