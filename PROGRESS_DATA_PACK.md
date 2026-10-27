# PROJECT PROGRESS DATA PACK

**Project:** Adversarial Self-Play for Code Bug Detection
**Audit date:** 2026-10-06, 22:39 UTC
**Repository:** `/home/amartyaranjan888/projects/rl`. Not yet committed to git (branch `master`, no commits).

**How this audit was done.**
- Every claim below was checked against files in the repository, test runs, or command output produced during the audit.
- **No GPU work was done for the audit.** Two CPU-only audit scripts were added and run: `scripts/audit_gate.py` and `scripts/audit_g6.py`.
- Two existing probe outputs that had only been saved to temporary files were copied, unchanged, into `logs/` so they can be cited.
- The original research specification (`2rlproj.pdf`) is **not in the repository**; the design is captured in `PLAN.md`.
- There is no project README.

**Evidence levels used throughout:**

| Label | Meaning |
|---|---|
| **Implemented** | Code exists and was read during the audit |
| **Unit-tested** | Covered by passing pytest tests |
| **CPU end-to-end** | Exercised inside the full co-training loop on CPU (tiny random model) |
| **GPU-validated** | Ran on the real GPU with a real model, with logged output |
| **Experimental evidence** | Produced a research measurement (a metric on real data) |

---

## A. Executive status

**Overall completion: ≈65%** (range 58–72% depending on weighting; see §B). Confidence: **high** for component status (verified against code, tests and logs); **medium** for the percentage, because the category weights are a judgement call.

**Major completed components:**
- **Data:**
  - training corpus of 24,048 + 952 test-verified functions;
  - held-out set of 3,500 real bug pairs (693 dev / 2,807 test) from two public sources;
  - 836-mutant synthetic validation set.
- **Environment:**
  - 10-class bug taxonomy with a deterministic AST-diff classifier;
  - format-preserving mutation engine;
  - six-check validity gate (G1–G6) including a unit-test behaviour check;
  - sandboxed test runner.
- **RL implementation:**
  - GRPO (advantages, clipped surrogate, k3 KL);
  - two LoRA adapters on one frozen base;
  - vLLM multi-LoRA rollout;
  - co-training loop with opponent pools, diversity tracker and curriculum;
  - checkpoint/resume; supervised fine-tuning (SFT) warmup.
- **Measured results:**
  - linter and Qwen2.5-Coder-1.5B zero-/few-shot baselines on real-dev (1,386 examples) and synthetic-val (1,672 examples);
  - Gemini on a 30-example sample;
  - injector readiness before and after warmup: gate pass rate 0.0% → 48.0%.
- **Engineering:** 126 tests, all passing; GPU smoke run of the full loop (0.5B model, 6 steps).

**Major blockers:**
1. **GPU memory.** The shared NVIDIA L4 has ~6.3 GB free. The co-training configuration is estimated to need ~19–21 GB; that is a planning estimate, not a measurement. **No 1.5B co-training run has been executed.**
2. **Code not yet written:**
   - A4 PPO (critic and generalized advantage estimation, GAE);
   - A5 group-size configs;
   - the validation against TRL (Hugging Face's reference RL library);
   - the supervised (SFT) detector baseline;
   - the figures script and the Dockerfile.
3. **External baselines incomplete.** The full Gemini run is waiting on a model/cost decision, and there is no Anthropic API key for Claude.

---

## B. Completion matrix

### B.1 Component matrix

| Component | Status | Evidence | Completion % |
|---|---|---|---|
| Research design and plan | Implemented (document) | `PLAN.md`; arm configs `configs/arms/*.yaml` | 85 |
| Taxonomy v1.0 (10 classes) | Implemented, unit-tested, used on real data | `src/advbug/taxonomy.py`; fingerprint `d6b0e82f9b10` in all manifests | 100 |
| Mutation engine (libcst) | Implemented, unit-tested, used to build 836 + 2,628 gated examples | `mutation/operators.py`; `tests/test_operators.py` (42) | 100 |
| AST classifier | Implemented, unit-tested; labels 3,426/3,500 real bugs | `mutation/classify.py`; `tests/test_classify.py` (31); `data/heldout/manifest.json` | 100 |
| Sandbox | Implemented, unit-tested; used for 25k-function corpus verification | `sandbox.py`; `tests/test_sandbox.py` (6) | 95 |
| Training corpus | Built | `data/corpus/manifest.json` | 100 |
| Held-out real bugs | Built (BugsInPy, a planned optional extra, not added) | `data/heldout/manifest.json` | 95 |
| Synthetic-val | Built | `data/synthetic/manifest.json` | 100 |
| Prompts and parsers | Implemented, unit-tested, GPU-validated (1.5B detector format failure 0.1%) | `game/prompts.py`; `results/base/dev.metrics.json` | 100 |
| Validity gate G1–G6 | Implemented, unit-tested, validated on real model outputs and 3,952 AST-engine candidates | `game/gate.py`; `logs/probe_injector_*.log`; `logs/audit_gate_funnel.json` | 100 |
| Rewards (detector, injector, diversity, subtlety) | Implemented, unit-tested, CPU end-to-end; GPU only in the 0.5B smoke run (detector side) | `game/rewards.py`; `tests/test_gate_rewards.py` | 90 |
| LoRA policy | Implemented, unit-tested, GPU-validated (SFT warmup, 0.5B smoke) | `rl/policy.py`; `tests/test_rl_core.py` | 95 |
| vLLM rollout | Implemented, GPU-validated (evals, probes, smoke) | `rl/rollout.py`; `logs/eval_base_*.log` | 95 |
| HF/CPU rollout | Implemented, unit-tested (test use only) | `rl/rollout.py`; `tests/test_rl_core.py` | 100 |
| GRPO | Implemented, unit-tested, CPU end-to-end, GPU 0.5B smoke; **no learning curve yet** | `rl/grpo.py` | 80 |
| Co-training loop | Implemented, CPU end-to-end (all Tier 1 + A6/A7 arms), GPU 0.5B smoke | `rl/loop.py`; `tests/test_loop_smoke.py`; `runs/smoke_05b-s0/` | 80 |
| Opponent pools | Implemented, CPU end-to-end, GPU snapshots in smoke run | `checkpoint.pt` pools `{injector:[4,6], detector:[4,6]}` | 80 |
| Curriculum (A7) | Implemented, CPU end-to-end only | `rl/loop.py:165-178`; smoke test `overrides3` | 70 |
| SFT warmup | Implemented, GPU-validated with experimental effect | `rl/sft.py`, `scripts/warmup_injector.py`, `runs/warmup_injector/manifest.json` | 100 |
| Evaluation metrics | Implemented, unit-tested, used; **ECE not wired into any script** | `eval/metrics.py`; `scripts/eval.py` | 85 |
| Baselines | Linter and base zero-/few-shot complete; Gemini 30-example sample only; Claude not run; SFT baseline not implemented | `results/*/` | 50 |
| Statistical testing | Implemented and unit-tested; **no analysis script uses it yet** | `eval/stats.py`; `tests/test_stats_baselines.py` | 60 |
| Configuration system | Implemented; exercised by all runs and tests | `config.py`, `configs/` | 100 |
| Checkpointing / resume | Implemented, CPU end-to-end (resume asserted), GPU checkpoint written | `tests/test_loop_smoke.py::test_adversarial_steps_pools_and_resume`; `runs/smoke_05b-s0/checkpoint.pt` | 90 |
| Logging | Implemented (JSONL; W&B optional, unused) | `runs/smoke_05b-s0/metrics.jsonl` | 90 |
| Tests | 126/126 passing | pytest run, §K | 90 |
| GPU smoke test | 0.5B, 6 steps, end-to-end | `runs/smoke_05b-s0/`, `logs/smoke_05b.log` | 100 (as a smoke test) |
| A4 PPO | **Not implemented** | none | 0 |
| A5 group-size sweep | **Configs not written** (loop supports any K) | none | 10 |
| TRL comparison | **Not implemented** | none | 0 |
| Tier 1/2/3 runs | **Not run** | none | 0 |
| Figures | **Not implemented** | none | 0 |
| Docker / release | **Not implemented** | none | 0 |

### B.2 Weighted completion model

| Category | Weight | Done | Weighted | Evidence | Remaining |
|---|---|---|---|---|---|
| Research design | 10% | 85% | 8.5 | `PLAN.md`; taxonomy and arm configs fixed | Confirm taxonomy deviations and the A7 definition; freeze hyperparameters after the pilot |
| Data pipeline | 15% | 95% | 14.25 | All three manifests with hashes | Optional BugsInPy; Kaggle dataset (link pending) |
| Mutation + validity environment | 15% | 95% | 14.25 | 42 + 31 + 17 tests; real-output gate statistics; G6 repeatability audit | G6 flakiness mitigation; `none_guard` sparsity in the corpus |
| RL implementation | 20% | 70% | 14.0 | GRPO/loop/pools/curriculum/SFT implemented and tested; 0.5B GPU smoke | PPO (A4), A5 configs, TRL validation, first 1.5B run |
| Baselines / evaluation | 15% | 55% | 8.25 | 8 metric files in `results/` | Full Gemini, Claude, SFT baseline, ECE integration, final test-set evaluation |
| Testing / engineering | 10% | 85% | 8.5 | 126 passing tests; CPU end-to-end per arm | Script tests, coverage measurement, API-path tests |
| Experiments / results | 10% | 10% | 1.0 | Baselines and injector readiness only | All co-training runs (Tier 1–3) and the analysis |
| Documentation / reproducibility | 5% | 55% | 2.75 | `PLAN.md`, `PROGRESS.md`, manifests, `uv.lock`, configs | README, Dockerfile, git history, release |
| **Total** | 100% | | **71.5%** | | |

**Sensitivity.** The research contribution is the ablation study, so a results-weighted scheme is also defensible: experiments at 30%, the other categories scaled down proportionally. That gives (70.5/90) × 70 + 0.10 × 30 = **57.8%**.

**Recommended headline: ≈65%, range 58–72%.** Engineering and data are close to complete. The experimental core, the co-training runs and their ablations, has not started.

---

## C. Dataset results

### C.1 Training corpus (`data/corpus/`)

| Item | Value |
|---|---|
| Source | KodCode/KodCode-V1 (Hugging Face, `refs/convert/parquet`), 15 parquet shards |
| Raw rows scanned | **484,097** |
| Rejected by static filters | top-level statement 84,843; length 48,941; few tests 37,850; non-stdlib import 19,119; long line 15,566; function count 2,638; test import 303; syntax 209; tests too long 5; tabs 1 |
| Exact duplicates removed (AST hash) | **106,054** |
| Unique candidates after static filters | 168,568 (`data/corpus_build.log`, "static: 484097 rows -> 168568 unique candidates") |
| Near-duplicates removed (MinHash, Jaccard ≥0.8) | **352** |
| Low mutation diversity (<3 applicable classes) | 1,004 |
| Failed own tests in sandbox | **442** |
| Flaky (passed once, not twice) | **9** |
| Kept | **25,000** (target reached; the scan stopped there) |
| Final split | **train 24,048 / val 952** (stable hash of the id, bucket 0 of 25 → val) |
| Constraints | 8–40 non-blank lines; lines ≤120 characters; 1–3 top-level functions plus stdlib imports only; ≥3 zero-argument tests; test file ≤6,000 characters; tests pass twice within 3 s |
| Subsets | Filter 6,016 · Taco 4,270 · Prefill 2,179 · Leetcode 2,127 · Code_Contests 2,025 · Codeforces 1,942 · Algorithm 1,849 · Data_Structure 1,564 · Apps 1,229 · Evol 680 · Package 577 · Docs 542 |
| Build time | 579 s (`build_seconds`) |
| sha256 | train `9bd84667…4dd25`; val `289032be…eeef4` |

### C.2 Held-out real bugs (`data/heldout/`)

| Item | Value |
|---|---|
| Sources | PyPIBugs (Microsoft/BugLab; Hugging Face mirror `Nadav-Timor/PyPiBugs`, 2,436 fix records) and TSSB-3M (Zenodo record 5845439; class-stratified sample of 4,000 records whose statement-level change maps to the taxonomy) |
| Pairs (buggy + fixed function) | **3,500**: PyPIBugs 1,766, TSSB 1,734 |
| Split | **dev 693 pairs (1,386 examples) / test 2,807 pairs (5,614 examples)**. Stable hash of the pair id, so pairs never straddle splits |
| Classes (all pairs) | variable_misuse 1,037 · negation 404 · argument_swap 367 · boundary_shift 343 · comparison_op 273 · wrong_return 246 · arithmetic_op 222 · type_conversion 199 · none_guard_omission 192 · boolean_op 143 · **other 74** |
| PyPIBugs classes | variable_misuse 873, negation 240, argument_swap 219, boundary_shift 154, comparison_op 98, arithmetic_op 70, other 63, boolean_op 49 |
| TSSB classes | wrong_return 246, type_conversion 199, none_guard_omission 192, boundary_shift 189, comparison_op 175, negation 164, variable_misuse 164, arithmetic_op 152, argument_swap 148, boolean_op 94, other 11 |
| Locations | call_arg 1,185 · condition 949 · statement 628 · return 600 · loop_header 64 |
| "other" | Real bugs that the classifier cannot map to a taxonomy class; scored for localization only |
| Dropped | TSSB: unparsable (mostly Python 2) 1,106; duplicate 574; file unavailable 352; function too long (>80 lines) 226; outside a function 8. PyPIBugs: too long 314; unavailable 250; duplicate 56; outside a function 49; unparsable 1 |
| Leakage checks | MinHash near-duplicate check against the full corpus: **0 found**; cross-source duplicates removed by identical buggy code |
| Labelling | `classify(fixed_function, buggy_function)`, the same classifier the gate uses. Bug lines are the buggy-file lines the fix touched; for a pure insertion, the line where the insertion goes |
| sha256 | dev `ef69b692…2edb5`; test `9760950b…e9d4efb` |
| Not checked (cannot be) | Contamination of the base model's pretraining data |

PyPIBugs alone covers only 8 of the 10 classes (no `wrong_return`, `type_conversion` or `none_guard_omission`); TSSB fills those gaps.

### C.3 Synthetic-val (`data/synthetic/val.jsonl`)

| Item | Value |
|---|---|
| Built from | 952 corpus val functions; one class-balanced AST-engine mutant each, full gate (G2–G6) |
| Mutants accepted | **836** (the 116 rejected were all G6 equivalent mutants; §F.3) |
| Clean originals | 836 (one per mutant) → **1,672 examples** |
| Classes | wrong_return 103 · negation 101 · variable_misuse 100 · arithmetic_op 92 · type_conversion 87 · boolean_op 85 · boundary_shift 84 · comparison_op 84 · argument_swap 79 · none_guard_omission 21 |
| Locations | condition 288 · statement 185 · return 182 · call_arg 123 · loop_header 58 |
| sha256 | `352cde17…c96f357` |

---

## D. Baseline results

All numbers below are from `results/<name>/<split>.metrics.json`, key `overall`.
- *Accuracy*: an exact line plus the correct type for an in-taxonomy bug, or "no bug" on a clean function.
- *Line exact*: share of buggy examples whose predicted line is one the bug touched.
- *Type accuracy*: on in-taxonomy bugs.
- *FPR*: share of clean functions flagged as buggy.

| Model | Dataset | n | Accuracy | Line exact | Type acc. | Macro-F1 | FPR | Format failure |
|---|---|---|---|---|---|---|---|---|
| pyflakes 4.0.2 | real-dev | 1,386 | 0.072 | 0.091 | 0.000 | 0.020 | 0.857 | 0.000 |
| pyflakes 4.0.2 | synthetic-val | 1,672 | 0.488 | 0.029 | 0.000 | 0.062 | 0.024 | 0.000 |
| Qwen2.5-Coder-1.5B zero-shot | real-dev | 1,386 | 0.370 | 0.019 | 0.020 | 0.073 | 0.268 | 0.001 |
| Qwen2.5-Coder-1.5B few-shot (2 examples) | real-dev | 1,386 | 0.013 | 0.157 | 0.119 | 0.055 | 0.999 | 0.000 |
| Qwen2.5-Coder-1.5B zero-shot | synthetic-val | 1,672 | 0.465 | 0.016 | 0.011 | 0.071 | 0.074 | 0.000 |
| Gemini 3.1 Pro preview (default thinking) | real-dev, first 30 rows | 30 | 0.667 | 0.800 | 0.800 | 0.816 | 0.467 | 0.033 |
| Gemini 3.1 Pro preview (`low` thinking) | real-dev, first 30 rows | 30 | 0.800 | 0.667 | 0.667 | 0.670 | 0.067 | 0.000 |
| Gemini 3.8 Flash | real-dev, first 30 rows | 30 | 0.833 | 0.867 | 0.867 | 0.926 | 0.200 | 0.000 |
| Claude (claude-opus-5-5) | — | — | **not run** (no API key) | | | | | |
| SFT detector on static mutations | — | — | **not implemented** | | | | | |
| A1-uniform (GRPO on static mutations) | — | — | **not run** (needs GPU) | | | | | |

**Further base zero-shot numbers on real-dev:**
- line within ±1: 0.048; within ±3: 0.098;
- joint (exact line and type): 0.003;
- flags a bug on 25.7% of buggy inputs.

Its prediction distribution (`results/base/dev.predictions.jsonl`): `no_bug` 1,022 of 1,386; most common predicted line 10 (162 times). Few-shot: never `no_bug`; line 10 predicted 294 times.

**By source on real-dev:**

| Baseline | PyPIBugs: accuracy / line / type / FPR | TSSB: accuracy / line / type / FPR |
|---|---|---|
| Base zero-shot | 0.385 / 0.020 / 0.014 / 0.242 | 0.354 / 0.018 / 0.027 / 0.297 |
| Base few-shot | 0.007 / 0.121 / 0.072 / 1.000 | 0.019 / 0.196 / 0.167 / 0.997 |
| Linter | 0.060 / 0.076 / 0 / 0.882 | 0.085 / 0.107 / 0 / 0.831 |

**Gemini cost data** (`results/gemini-*/dev.gemini_cache.jsonl`). Each run used the same 20,348 input tokens.

| Model setting | Output tokens | Thinking tokens | Finish reasons |
|---|---|---|---|
| Pro, default thinking | 1,045 | 164,398 | 29 STOP, 1 MAX_TOKENS |
| Pro, `low` thinking | 346 | 13,983 | 30 STOP |
| 3.8 Flash | 377 | 48,319 | 30 STOP |

The exact `model_version` is recorded per answer.

**Caveat:** the 30-row Gemini sample is entirely PyPIBugs (15 buggy + 15 clean). It covers no TSSB-only classes and is far too small to rank models.

**What each baseline shows:**
- **Linter:** static checks don't find semantic bugs. Functions extracted from their modules produce many undefined-name warnings, hence the 0.857 FPR on real code.
- **Base zero-shot:** this is the RL starting point. It mostly answers `no_bug` and has essentially no localization ability.
- **Base few-shot:** the prompt flips the model to always flagging a bug. Its behaviour is driven by priors (the line-10 preference), not understanding.
- **Gemini:** a strong frontier model can localise most of these real bugs on this small sample. That gives an upper reference point, at a significant thinking-token cost.

---

## E. Injector results

### E.1 Readiness probe

`scripts/probe_injector.py`: 32 val functions (seed 0) × 8 samples = **256 samples**, temperature 1.0, ≤128 tokens, full gate G1–G6, vLLM at 24% GPU memory in eager mode.

| Metric | Base Qwen2.5-Coder-1.5B | After format warmup |
|---|---|---|
| Gate pass rate | **0.000** | **0.480** (123/256) |
| G1 format | 0.574 | 0.031 |
| G2 size | 0.164 | 0.098 |
| G3 compile | 0.160 | 0.039 |
| G4 artifact | 0.004 | 0.000 |
| G5 label | 0.098 | 0.324 |
| G6 equivalent | 0.000 | 0.027 |
| Functions with ≥1 valid mutant | **0/32** | **32/32** |
| Valid classes produced | none | negation 31, comparison_op 20, wrong_return 20, variable_misuse 13, arithmetic_op 11, boundary_shift 11, type_conversion 6, boolean_op 6, argument_swap 5 (9 of 10; no none_guard_omission) |

Evidence: `logs/probe_injector_base.log`, `logs/probe_injector_warm.log`.

### E.2 Format warmup (SFT)

`scripts/warmup_injector.py`; `runs/warmup_injector/manifest.json`; `logs/warmup_injector.log`.

| Item | Value |
|---|---|
| Data | 3,000 training-corpus functions (seed 0) → **2,628 gated AST-engine examples**, written as injector answers by `format_injector_output` |
| Classes | negation 324, wrong_return 321, variable_misuse 319, arithmetic_op 311, boolean_op 272, comparison_op 267, boundary_shift 264, argument_swap 259, type_conversion 251, none_guard_omission 40 |
| Training | LoRA (r=32, α=64, all linear layers) on the injector adapter only; AdamW, learning rate 1e-4; batch 16 with micro-batches of 4; 1 epoch (165 optimizer steps); gradient clip 1.0; cross-entropy on completion tokens only |
| Loss | first steps 0.589, 0.398, 0.326, 0.322, 0.281 → last steps 0.132, 0.113, 0.117, 0.117, **0.107** |
| Adapter | `runs/warmup_injector/injector/` (`adapter_model.safetensors`, 141 MB, 36,929,536 float32 parameters, 392 tensors) |
| Use | `configs/base.yaml: injector.init_adapter`, so every arm starts from it. A1-frozen's frozen injector is this warmed adapter. |

**What changed.** Without the warmup, every injector sample is rejected, so every GRPO group has identical rewards (−1). The advantages are all zero and the injector cannot learn. After the warmup, 32 of 32 functions produce at least one valid mutant per 8 samples, so groups have reward variance. The dominant remaining failure is a mismatched class claim (G5, 32.4%), which the RL reward penalises (−1).

**Caveat:** the warmup biases the injector's starting distribution towards AST-engine mutation styles.

### E.3 Injector outputs actually logged

Only 3 samples per probe were printed; per-sample outputs were not saved.

| Probe | Claimed class | Model output (abridged) | Gate verdict |
|---|---|---|---|
| Base | arithmetic_op | rewrote lines 7–10 into a 5+ line block | **G2 size** |
| Base | arithmetic_op | `if k < 1 or k > len(scores) + 1:  # Changed the boundary shift` | **G4 artifact**: a comment that gives the answer away |
| Base | boundary_shift | tags preceded by stray text; replaced lines 2–4 with a single unrelated line | **G5 label** |
| Warmed | wrong_return | replaced line 5 with `if not scores:` | **G5 label** |
| Warmed | arithmetic_op | `return sorted_scores[k+1]` | **VALID** |
| Warmed | negation | `if scores or k < 1 or k > len(scores):` (dropped `not`) | **VALID** |

### E.4 Representative gated mutants, with AST class and test outcomes

These are AST-engine mutants, not injector-policy outputs. They come from the CPU replay in `scripts/audit_gate.py` and are recorded in `logs/audit_gate_examples.txt`.

| Function | Original → mutation | Claimed | AST class | Gate | Tests: original → mutant |
|---|---|---|---|---|---|
| Algorithm_10037_I | `return dp[0][n - 1]` → `return None` | wrong_return | wrong_return | ACCEPTED | 6/6 pass → 6 fail |
| Algorithm_10128_I | `range(1, n)` → `range(n, 1)` | argument_swap | argument_swap | ACCEPTED | 4/4 → 3 fail |
| Algorithm_10446_I | `return list(buffer)` → `return tuple(buffer)` | type_conversion | type_conversion | ACCEPTED | 8/8 → 6 fail |
| Algorithm_10618_I | `k = k % n` → `k = k // n` | arithmetic_op | arithmetic_op | ACCEPTED | 5/5 → 3 fail |
| Algorithm_10891_I | `return False` → `return True` | negation | negation | ACCEPTED | 6/6 → 1 fail |
| Algorithm_11430_I | `words[::-1]` → `s[::-1]` | variable_misuse | variable_misuse | ACCEPTED | 7/7 → 7 fail |
| Algorithm_11533_I | `matrix[j][i]` → `matrix[j][i - 1]` | boundary_shift | boundary_shift | ACCEPTED | 6/6 → 4 fail |
| Algorithm_11707_I | `if k == 0 or …` → `if k >= 0 or …` | comparison_op | comparison_op | ACCEPTED | 8/8 → 6 fail |
| Algorithm_12621_I | `… == 0 or … == num_rows - 1` → `and` | boolean_op | boolean_op | ACCEPTED | 7/7 → 4 fail |
| Apps_13694_I | `dp[i-1][j] or dp[i-1][j-arr[i-1]]` → `dp[i-1][j]` | none_guard_omission | none_guard_omission | ACCEPTED | 9/9 → 7 fail |
| Algorithm_10486_I | `-> List[int]` → `-> List[int + 1]` (type annotation) | boundary_shift | boundary_shift | **REJECTED G6** | 9/9 → **9/9 still pass** (equivalent mutant) |
| Docs: Python310_31584_I | multi-line `return {...}` → `return None` | wrong_return | wrong_return | **REJECTED G2** | 5/5 → 5 fail (rejected for size) |

**Taxonomy limitation.** Apps_13694_I shows that the `x or default` → `x` rule labels removal of a non-None `or` fallback as `none_guard_omission`.

---

## F. Gate results

### F.1 Checks

Code: `src/advbug/game/gate.py`; configuration `GateConfig(max_edit_lines=3, max_edit_chars=120, max_line_chars=120)`.

**G1 format** (`Gate._parse`, l.117–126)
- **What it checks:** `parse_injector` finds `<type>`, `<lines>A-B</lines>` and `<new>…</new>`; `apply_action` gets a valid range (1 ≤ A ≤ B ≤ number of lines); the claimed class is in the taxonomy.
- **Why:** the detector must be scored against a parseable claim. G1 stays on even in the A3 arm (gate disabled).
- **Prevents:** free-form outputs and invented classes.
- **Tests:** 3 cases in `test_gate_checks`.
- **Real rejections:** base probe 57.4%; warmed probe 3.1%.

**G2 size** (`static_checks`, l.86)
- **What it checks:** the edit is non-empty, changes ≤3 lines (`edited_line_count`), and inserts plus deletes ≤120 characters (`edit_chars`, character-level difflib).
- **Why:** keeps mutations minimal and single-cause.
- **Prevents:** rewriting the function, and no-op edits.
- **Tests:** no-op case.
- **Real rejections:** base 16.4%; warmed 9.8%; AST replay 3 of 3,000.

**G3 compile** (l.88–91)
- **What it checks:** `compile(mutant)`.
- **Why:** a syntactically broken mutant is a trivial detection cue.
- **Prevents:** reward from unparseable code.
- **Tests:** a broken-expression case.
- **Real rejections:** base 16.0%; warmed 3.9%.

**G4 artifact** (`_artifact`, l.67–77)
- **What it checks:** no comment added or changed (tokenize-based comment multiset), and no new line with a tab, trailing whitespace or >120 characters.
- **Why:** these are detection shortcuts unrelated to the bug.
- **Prevents:** the injector or detector exploiting formatting cues.
- **Tests:** two cases (added `# bug`, changed comment).
- **Real rejections:** base 0.4%, including the logged `# Changed the boundary shift`.

**G5 label** (l.94–96)
- **What it checks:** `diagnose(original, mutant).bug_class == claimed`.
- **Why:** labels must be correct and the edit single-cause.
- **Prevents:** mislabelled claims (which would corrupt the detector's targets), multi-edit mutations, and edits outside the taxonomy.
- **Tests:** 4 cases (wrong claim, operand swap, two non-taxonomy rewrites).
- **Real rejections:** base 9.8%; warmed 32.4%.

**G6 behaviour** (`_behaviour`, l.128–137)
- **What it checks:** the mutant is run against the function's own unit tests in the sandbox; it is rejected if `RunResult.all_passed`.
- **Why:** an *equivalent mutant* can never be detected but would earn maximum fool reward. This is the central reward hack.
- **Prevents:** reward from behaviour-preserving edits.
- **Tests:** `test_equivalent_mutant_rejected_by_tests` (`x * 1` → `x // 1`).
- **Real rejections:** warmed 2.7%; AST replay 117/952 and 362/3,000.

**How G6 establishes non-equivalence.**
- **The original passes.** It passed all of its tests twice at corpus-build time (`data/corpus.py: build_corpus`).
- **The mutant is tested in a fork.** `sandbox.run_tests` forks a warm spawn-worker. The child:
  - applies rlimits: CPU = timeout + 1 s, address space = current mapping + 1 GiB, file size 16 MiB, no core dumps;
  - uses a private temporary directory;
  - blocks socket, subprocess, kill and delete calls;
  - executes the mutant as module `solution`, then the test file, with the solution's names pre-loaded.
- **What gets run.** Every zero-argument `test*` function and every `Test*` class method. Parametrised and fixture-based tests are skipped.
- **Result.** The child reports `passed`/`failed`. The parent enforces a 2 s wall-clock timeout (`SandboxPool` default) and kills the process group on expiry.
- **The verdict:**
  - accepted if ≥1 test fails, the module fails to load, or the run times out;
  - rejected as `G6_equivalent` if every test passes.
- **Order.** G6 runs only on candidates that passed G1–G5, so injector-written code executes only once it is known to be a single taxonomy edit of corpus code.
- **Limitation.** This is equivalence *under the tests*. A behaviour change the tests miss is rejected as "equivalent", which is conservative. A timeout counts as a behaviour change.

### F.2 G6 outcome types and repeatability

`scripts/audit_g6.py` → `logs/audit_g6.json`: 952 synthetic-val candidates reaching G6, run twice.

| Outcome | Run 1 | Run 2 |
|---|---|---|
| Failing assertions (accept) | 820 | 822 |
| All tests pass (reject) | 117 | 116 |
| Timeout (accept) | 8 | 7 |
| Error on load (accept) | 7 | 7 |
| **Verdict agreement** | **99.89%** (1 flip in 952: `Filter_25248_I`, all-pass ↔ failing, a flaky test) | |

- Of accepted mutants, **98.2% (820/835) were accepted because tests genuinely failed**; timeouts account for 8.
- Failing tests per accepted mutant: 1: 84, 2: 98, 3: 138, 4: 152, 5: 155, 6: 111, 7: 51, 8: 24, 9: 5, 10: 1, 11: 1.
- **Replication:**
  - synthetic-val, replayed with the same seed: 836 accepted, identical to the manifest (0 differing verdicts in the second replay; one earlier replay gave 835);
  - warmup: **2,635 replayed vs 2,628 recorded** (7 more accepted). Generation and G2–G5 are deterministic, so G6 caused the difference. **G6 is slightly non-deterministic** (flaky tests, timeouts under CPU load).

### F.3 Rejections of AST-engine mutants

`logs/audit_gate_funnel.json`:
- **Synthetic-val:** 952 candidates → 835/836 accepted; rejections were **all G6** (117). By claimed class: argument_swap 24, boundary_shift 20, boolean_op 19, comparison_op 19, type_conversion 16, arithmetic_op 11, variable_misuse 3, negation 2, none_guard 2, wrong_return 1.
- **Warmup:** 3,000 candidates → 2,635 accepted; G2 3, G6 362. By class: type_conversion 74, argument_swap 69, boundary_shift 62, comparison_op 59, boolean_op 54, arithmetic_op 17, none_guard 10, variable_misuse 9, wrong_return 7, negation 4.
- **Meaning:** about 12% of syntactically valid, correctly labelled single edits are behaviour-preserving under the tests. Without G6 those would be uncatchable "bugs" with maximum fool reward.

---

## G. RL implementation

### G.1 Formulation

Code: `src/advbug/rl/loop.py`, `CoTrainer.step`, l.108–220.

**Structure.**
- **Game:** a two-player, general-sum game played as a contextual bandit. Each episode is a single decision; there are no multi-step transitions and no discounting.
- **Both players are policies over token sequences:** π_θ is the base model with the `injector` LoRA, and π_φ is the base model with the `detector` LoRA. The base is Qwen2.5-Coder-1.5B-Instruct, frozen in bf16.
- **Trainable parameters:** 36.9M per adapter (LoRA r=32, α=64, dropout 0, all linear modules).

| Element | Injector | Detector |
|---|---|---|
| Observation | Chat prompt: system "expert Python programmer and code reviewer" + taxonomy (10 classes with descriptions) + rules (≤3 consecutive lines, keep indentation, no comment changes, must still run, must change behaviour) + the clean function with line numbers (`injector_messages`) | System prompt + taxonomy + line-numbered code (a valid mutant, a mutant from a past injector, or a clean corpus function) + answer format (`detector_messages`) |
| Action | ≤128 tokens; parsed as `<type>C</type><lines>A-B</lines><new>…</new>` | ≤32 tokens; parsed as `<line>N</line><type>C</type>` or `<type>no_bug</type>` |
| Sampling | vLLM, temperature 1.0; K=8 per function; B=16 functions per step (128 samples) | vLLM, temperature 1.0; K_D=4 per input |
| Transition | Deterministic environment: parse → apply edit (re-indent if needed) → gate G1–G6 → mutant | None (terminal) |
| Reward | `injector_reward` (§I) | `detector_reward` (§I) |
| GRPO group | The K samples for one function | The K_D answers for one input |
| Reference policy | Base model with adapters disabled (`Policy.completion_logprobs(adapter=None)`) | Same |

### G.2 One co-training step (defaults from `configs/base.yaml`)

1. **Sample injector inputs.** Draw B=16 corpus functions (seed-driven RNG); the injector generates K=8 samples each; `Gate.check` runs on all 128.
   - A1-frozen: the frozen initial injector is used instead.
   - A1-uniform: B·K=128 functions go to the AST engine instead (`uniform_mutants`).
2. **Pool data (if pools are on and non-empty).** Take round(0.25 × 64) = 16 extra functions, sample one mutant each from a random past injector, and gate them. They are used as detector data only.
3. **Detector pass.** Detector inputs are all valid current mutants, the pooled mutants, and 16 clean functions. The detector answers K_D=4 times per input. Rewards are computed per answer, plus binary correctness.
4. **Fool rate per current mutant.** f = 1 − mean(correct). If pools are on, a random past detector answers greedily once and f' = 0.7·f + 0.3·(1 − past_correct).
5. **Injector reward matrix [B, K].** Initialised to −1; valid samples get `injector_reward`. Curriculum (A7) modifies this (§I). The diversity tracker is updated afterwards.
6. **Detector training groups.** Up to 64 mutant inputs (current mutants plus up to 16 pooled) plus the 16 clean inputs, each with K_D answers.
7. **Updates.** `_update("injector", …)` (adversarial arm only), then `_update("detector", …)`: one GRPO optimizer step each.
8. **Publish and snapshot.** Save both adapters as new versions; vLLM serves them through a new `LoRARequest`, with the newest two versions kept on disk. Every 20 steps, snapshot to the pools (keep 10).
9. **Log dynamics.** Injector class entropy, number of classes, fool rate, on-policy synthetic accuracy (and its 0.9-EMA, which drives the curriculum), clean accuracy, detector format failures, gate check rates, and per-player reward mean/std, zero-variance group fraction, advantage variance, gradient norm, KL, trained-sequence count.

**Evaluation** (`scripts/train.py`): every 25 steps, greedy decoding of the current detector on fixed subsets (the first 400 real-dev and first 400 synthetic-val examples, sorted by pair). Gaps are logged: `gap/synthetic_val_minus_real` and `gap/on_policy_minus_real`.

### G.3 Mechanisms

- **Opponent pools.** `CoTrainer._snapshot_pools` (l.96–104); used in `step` l.132–140 and l.157–163.
- **Curriculum (A7).** `step` l.165–178:
  - *phase A* (detector EMA accuracy < 0.5): fool term capped at 0.5;
  - *phase B*: valid mutants with f = 0 receive −1 and are excluded from detector training.
- **Format warmup:** `rl/sft.py`, `scripts/warmup_injector.py`.
- **Arms:**
  - `configs/arms/full.yaml` (all mechanisms on);
  - `a1_frozen` and `a1_uniform` (pools off);
  - `a2_off` (λ_div = 0, pools off);
  - `a3_off` (gate disabled except G1);
  - `a6_partial` (partial-credit detector reward);
  - `a7_curriculum`;
  - `smoke_05b` (tiny smoke configuration).
- **Not implemented:** `a4` (PPO), `a5` (K sweep).

---

## H. GRPO implementation

Code: `src/advbug/rl/grpo.py`, `CoTrainer._update` (`rl/loop.py` l.222–267), `Policy.completion_logprobs` (`rl/policy.py` l.82–119).

**Groups.**
- Injector: one group per function (K samples). Reward tensor `[B, K]` (`loop.py` l.168–176).
- Detector: one group per input (K_D answers). Tensor `[G, K_D]` (l.190).

**Advantage** (`group_advantages`, l.14–23), with the population standard deviation (`unbiased=False`) and ε = 1e-4:

  A_{g,k} = ( r_{g,k} − mean_k r_{g,k} ) / ( std_k r_{g,k} + 1e-4 )

**Zero-variance groups.** If all rewards in a group are equal, A = 0 for every member. These groups are counted (`zero_variance_fraction`), and with `skip_zero_advantage: true` their sequences are dropped from the update (l.238–239). Sequences with an empty completion are also dropped (l.236).

**Token log-probabilities** (`completion_logprobs`).
- Prompt and completion are concatenated and **left-padded**, so every completion ends at the last position.
- Position ids are recomputed as `cumsum(attention) − 1`, so they restart at 0 after padding.
- The model returns logits only for the last C_max+1 positions (`logits_to_keep`). These are cast to float32; log-softmax is applied and the probability of each realised token gathered.
- `mask` marks only completion tokens. **Prompt tokens never enter the loss.**
- This is tested against a full forward pass (`test_logprobs_match_full_forward`) and is invariant to padding (`test_logprobs_independent_of_padding`).

**Policy loss** (`policy_loss`, l.36–68). For each kept sequence i and completion token t, with ℓ_{i,t} = log π(o_{i,t} | s_i, o_{i,<t}):

  ρ_{i,t} = exp( ℓ_{i,t} − sg[ℓ_{i,t}] )   (old_logp = None, so the ratio has value 1 and gradient ∇ℓ)

  L^{PG}_{i,t} = −min( ρ_{i,t} A_i , clip(ρ_{i,t}, 1−0.2, 1+0.2) A_i )

  δ_{i,t} = ℓ^{ref}_{i,t} − ℓ_{i,t},  KL_{i,t} = e^{δ_{i,t}} − δ_{i,t} − 1  (k3 estimator, ≥0; reference = base model with adapters disabled, under `no_grad`)

  𝓛 = (1/N) Σ_i Σ_{t∈completion(i)} [ L^{PG}_{i,t} + β·KL_{i,t} ],  β = 0.01,  N = Σ_i |completion(i)| over the kept sequences of that player in that step

**Notes on the formulation.**
- **Aggregation is token-level across the whole batch** (`total_tokens`, l.243). Micro-batches each divide by the step-wide N, so accumulated gradients equal the full-batch loss. This differs from the original GRPO paper's per-sequence 1/|o_i| averaging; it is the "token-level" variant (as in DAPO).
- **The update is fully on-policy:** one optimizer step per rollout batch (μ=1), and **no stored old log-probabilities are needed**. The ratio is identically 1, so **the clip is inactive** and is logged as `clip_fraction`, which should be 0. The gradient reduces to −(1/N) Σ A_i ∇ℓ_{i,t} + β(1/N) Σ (1 − e^{δ}) ∇ℓ_{i,t}, i.e. REINFORCE with a group-normalised baseline plus a KL penalty.
- Samples come from vLLM serving the adapter version published after the previous update, which has the same weights as the training model. **There is no importance correction for numerical differences between vLLM and HF kernels.**
- Dropping zero-advantage sequences also drops their KL contribution.

**Optimisation.**
- **Optimizer:** AdamW per player (learning rate 1e-5, weight decay 0) over that player's LoRA parameters only. `Policy.use` turns gradients off for the other adapter; isolation is tested in `test_adapters_are_isolated`.
- **Schedule:** LambdaLR with linear warmup over 10 steps, starting at lr/10, then constant.
- **Gradient handling:** gradient clipping at norm 1.0; micro-batch 8 with gradient accumulation; gradient checkpointing on (non-reentrant).
- **Steps:** one optimizer step per player per co-training step.

**Numerical safeguards:** ε = 1e-4 in the standard deviation; float32 log-softmax; `max(total_tokens, 1)`; positions clamped at ≥0.

**Test evidence** (`tests/test_rl_core.py`):
- advantages and zero variance on a hand-computed example;
- loss value and normalisation; KL > 0 when the policy differs from the reference;
- one SGD step increases the log-probability of the positive-advantage completion and decreases the negative one.

---

## I. Reward implementation

Code: `src/advbug/game/rewards.py`; aggregation in `rl/loop.py` l.154–178. Defaults: λ_div = 0.1, λ_sub = 0.05, invalid penalty −1, pool weight w = 0.3.

**1. Detector reward, binary** (`detector_reward`, l.28–37)
- **Formula:** r = 1 if the prediction parses and either (y = no_bug and ŷ = no_bug) or (y ≠ no_bug, ŷ_class = y, and ŷ_line ∈ bug_lines). Otherwise 0. An unparseable answer scores 0.
- **Inputs:** the true class, the bug lines (the changed span in mutant coordinates) and the parsed prediction.
- **Range:** {0, 1}.
- **Purpose:** a verifiable joint localization and classification reward.
- **Failure mode:** sparse. A weak detector produces many zero-variance groups; in the smoke run, 50–75% of detector groups had zero variance.

**2. Detector reward, partial (A6)** (l.38–39)
- **Formula:** for a clean input, 1[ŷ = no_bug]. For a bug, 0.5·s_line + 0.5·1[ŷ_class = y], where s_line = 1 (distance d = 0), 0.5 (d ≤ 1), 0.25 (d ≤ 3), else 0, and d = min_b |ŷ_line − b|. Predicting `no_bug` on a bug gives d = ∞ and scores 0.
- **Range:** [0, 1] in steps of 0.125.
- **Purpose:** a denser signal; tests whether localization or classification is learned first.
- **Failure mode:** the detector may settle for "nearby line, right type".

**3. Injector fool rate** (`fool_rate`, l.42–43; loop l.155)
- **Formula:** f = 1 − (1/K_D) Σ_j 1[answer j correct (binary)].
- **Range:** {0, 0.25, 0.5, 0.75, 1} for K_D = 4.
- **Purpose:** the adversarial objective.
- **Failure modes:**
  - quantised and noisy at K_D = 4;
  - rewards any valid but undetectable bug, which is why G5/G6 matter;
  - always uses the binary criterion, even in A6.

**4. Historical opponent** (loop l.157–163, pools on)
- **Formula:** f' = (1 − w)·f + w·(1 − 1[past detector, greedy, correct]), w = 0.3.
- **Purpose:** discourages exploiting only the current detector (cycling).
- **Failure mode:** a single greedy answer is noisy.

**5. Diversity bonus** (`DiversityTracker`, l.50–70)
- **Formula:** b(c) = clip(−ln p̂(c) / ln 10, 0, 1), with p̂ initialised to 0.1 per class and updated after each step as p̂ ← 0.9·p̂ + 0.1·(batch frequency of c among valid mutants). The bonus uses p̂ *before* the current step's update.
- **Range:** [0, 1]; equals 1 for any class at or below the uniform rate.
- **Purpose:** counteracts mode collapse (A2).
- **Failure mode:** rarely used classes all get the maximum bonus, so it only penalises over-use. A class the injector can't produce validly (e.g. none_guard) earns nothing.

**6. Subtlety bonus** (`subtlety`, l.46–47)
- **Formula:** s = max(0, 1 − e/120), where e is the character-level edit size.
- **Range:** [0, 1).
- **Purpose:** prefers minimal edits.
- **Failure mode:** a small weight; can favour tiny edits that are easy to spot.

**7. Invalid sample** (`injector_reward`, l.76–77)
- **Formula:** −1 if any gate check fails. With the gate off (A3), only a G1 failure gives −1.
- **Purpose:** blocks reward hacking through invalid edits.
- **Failure mode:** when *all* samples are invalid there is no gradient. This was observed: base probe 0/256, 0.5B smoke run 100% G1. Mitigated by the warmup.

**8. Final injector reward** (l.73–79 and loop l.165–176)
- **Formula:** r_I = −1 if invalid. Otherwise r_I = min(f', cap) + 0.1·b + 0.05·s, with cap = 0.5 only in A7 phase A. In A7 phase B, a valid mutant with f = 0 gets −1 and is dropped from detector training.
- **Range:** [−1, 1.15].

---

## J. Smoke tests

### J.1 GPU end-to-end smoke run

`runs/smoke_05b-s0/` (`config.json`, `metrics.jsonl`, `checkpoint.pt`, `adapters/`); `logs/smoke_05b.log`.

| Item | Value |
|---|---|
| Model | Qwen2.5-Coder-0.5B-Instruct (vLLM + HF training, same process) |
| GPU | NVIDIA L4; vLLM share 0.12, eager mode; 0.99 GiB weights, 1.09 GiB KV cache (vLLM log) |
| Configuration | 6 steps, seed 0, B=4, K=4, K_D=4, train_mutants 16, train_clean 4, learning rate 1e-5, β=0.01, micro-batch 4, pools every 2 (keep 2), LoRA r=32/α=64 |
| Step time | 18.1, 12.5, 15.3, 20.5, 17.9, 20.5 s (mean 17.5 s; `time/step_seconds`) |
| Gate pass rate | **0.0 at every step** (G1 format 100%) |
| Injector updates | **none** (`injector/trained_sequences` = 0; all rewards −1) |
| Detector updates | 8, 8, 8, 4, 8, 8 sequences per step (clean inputs only); zero-variance groups 0.5–0.75; gradient norm 5.6–9.9 (clipped to 1.0); KL ≤0.001 |
| Detector format failures (training samples) | 0.75 → 0.69 → 0.75 → 0.94 → 0.81 → 0.88 |
| Evaluation (60 real-dev / 60 synthetic) | step 0: format failure 1.0 / 1.0, accuracy 0 / 0; after 3 updates: same; after 6 updates: format failure 0.90 / 0.967, accuracy 0.05 / 0.017 |
| Checkpoint | `checkpoint.pt` at step 6 with keys current, detector_acc, diversity, initial_injector, on_disk, optim, pools, rng, sched, step, torch_rng; pools hold versions 4 and 6 of each adapter |
| Adapters on disk | injector v0 (initial, protected), v4–v6; detector v4–v6 (older versions cleaned up) |
| Separate raw-output probe of 0.5B (`logs/probe_05b_raw_outputs.log`) | injector parseable 2/12, detector parseable 2/12 (e.g. bare `no_bug` without tags) |

**What it proves:**
- On the GPU, the loop runs end to end: vLLM generation with adapter hot-swapping, gate plus sandbox, rewards, GRPO updates (detector side), adapter publishing and cleanup, pool snapshots, periodic evaluation and checkpoint writing.

**What it does not prove:**
- It shows nothing about learning performance.
- The injector branch was never updated on the GPU.
- The 0.5B model cannot follow the output format, so it is unusable as a policy.

**Logging quirk:** the `step` field in `metrics.jsonl` is the index *before* the update. The evaluation rows labelled 2 and 5 were taken after 3 and 6 updates.

### J.2 CPU end-to-end loop tests

`tests/test_loop_smoke.py`, 5 tests. Each uses a tiny random Qwen2 model with the real tokenizer, a scripted generator (real AST mutants plus garbage for the injector, random answers for the detector), the real gate and sandbox, and real GRPO updates.
- **`full`:** 4 steps; asserts that the injector parameters changed, the pools filled, old adapters were cleaned up, and checkpoint/resume reproduces the adapter weights and step index.
- **Other arms:** A1-uniform, A1-frozen (injector unchanged), A3, and A7 + A6 each run 2 steps.

---

## K. Software tests

Command: `CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q`. Result: **126 passed, 0 failed, 0 skipped, 31.28 s** on CPU.

| File | Tests | Covers |
|---|---|---|
| `test_operators.py` | 42 | each class applies somewhere (10); 6 samples × 5 seeds: mutants compile, classifier agrees, ≤3 lines edited, untouched lines byte-identical, comments preserved (30); determinism; unparseable input |
| `test_classify.py` | 31 | 17 single-cause edits across all classes; argument swap (positional/keyword); none-guard only outside conditions; True/False; 6 rejections (comment-only, formatting-only, two changes, non-taxonomy, out-of-scope name, added statement); two independent edits; unparseable; priority; location tags; changed-span/deletion lines |
| `test_gate_rewards.py` | 17 | 13 gate cases spanning G1–G5 plus valid; equivalent mutant rejected by G6; gate-disabled A3; detector binary and partial rewards; injector reward, cap, diversity tracker |
| `test_rl_core.py` | 8 | advantages and zero-variance; loss value, normalisation, KL; padding invariance; full-forward alignment; adapter isolation; gradient direction; adapter save/reload; HF generator |
| `test_prompts_metrics.py` | 7 | line numbering; prompt content; detector/injector parsers; edit application and re-indent; metric values; F1 and ECE |
| `test_sandbox.py` | 6 | original passes, mutant fails, infinite-loop timeout, broken solution, blocked socket/subprocess/delete, memory bomb |
| `test_loop_smoke.py` | 5 | §J.2 |
| `test_heldout.py` | 5 | reverse-applying a replacement and an insertion; method extraction and labelling; drop reasons; splits keep pairs together |
| `test_stats_baselines.py` | 4 | McNemar, paired and hierarchical bootstrap, Holm; linter; answer cache |
| `test_format_injector.py` | 1 | injector-answer formatting round-trips for every class |

**Code size:** `src/advbug` 3,397 lines, `scripts/` 674 lines before this audit (two audit scripts were added since), `tests/` 897 lines.

**Missing tests:**
- no tests for the CLI scripts (`build_*`, `train.py`, `eval.py`, `run_queue.py`, `warmup_injector.py`, `probe_injector.py`);
- `VLLMGenerator` is exercised only on the GPU (no CI test);
- the Claude/Gemini request paths are untested (network);
- `static_reject_reason` and `read_tssb` have no direct unit tests;
- the curriculum phase switch and opponent sampling are not asserted beyond running;
- there is no test of a G6 timeout through the `Gate` itself (the sandbox timeout is tested);
- coverage is not measured.

---

## L. Hardware / compute

| Item | Status |
|---|---|
| GPU | NVIDIA L4, 23,034 MiB, driver 595.84 (`logs/gpu_info.txt`); CUDA 13.0 (torch cu130). **Shared with other users' services** |
| Free memory at audit time | **6,283 MiB** (16,284 MiB used by other processes; 0 from this project) |
| Planned co-training allocation | vLLM `gpu_memory_utilization` 0.35 (~8 GB) plus training (bf16 base ~3 GB, two float32 LoRA adapters with AdamW state, activations under gradient checkpointing). About **19–21 GB in total, a PLAN.md estimate, not measured** |
| Measured memory | 1.5B in vLLM: weights 3.05 GiB, KV cache 1.45 GiB at a 0.24 share (`logs/eval_base_splitdev.log`). 0.5B: weights 0.99 GiB, KV 1.09 GiB at 0.12 (`logs/smoke_05b.log`) |
| Batch sizes (full configuration) | Injector 16×8 = 128 samples per step; detector up to 80 inputs × 4 answers trained, more generated for fool rates |
| Step time | **Measured only for the 0.5B tiny configuration: mean 17.5 s.** 1.5B full-configuration step time **not measured**; PLAN.md's 1.5–2.5 min is an estimate |
| GPU runs actually executed | base-model evaluations (3); injector probes (base, warmed); injector SFT warmup; 0.5B smoke co-training (6 steps); 0.5B raw-output probe |
| **Not executed** | **Any 1.5B co-training run; any Tier 1/2/3 arm; the TRL comparison; the SFT detector baseline** |

---

## M. Remaining work

### Must do (for the planned contribution)

1. Secure about 19–21 GB of GPU memory, or adapt the configuration to less.
2. 1.5B smoke run (10 steps), then the pilot (Tier 1 × 1 seed × 100 steps); measure the step time; freeze the hyperparameters.
3. Validate GRPO: the TRL `GRPOTrainer` comparison (not written), or at minimum show that A1-uniform's detector accuracy rises above the base model's.
4. Tier 1 runs: `full`, `a1_frozen`, `a1_uniform`, `a2_off`, `a3_off` × 3 seeds.
5. Analysis script: dynamics curves, real-dev/test scoring of each final adapter, statistical tests (`eval/stats.py` exists).
6. Final real-test evaluation of every arm and baseline.

### Should do

1. A4 PPO (critic LoRA, value head, GAE) and its config.
2. A5 configs (K ∈ {2, 4, 16}, B·K = 128) plus the stationary control on `a1_uniform`.
3. Tier 3 runs (`a6_partial` and `a7_curriculum` exist).
4. Supervised detector baseline on static mutations.
5. Full Gemini dev run (choose a model), and Claude if a key becomes available.
6. Wire ECE (confidence from sampled answers) into `eval.py`.
7. Make G6 deterministic: retry an "all pass" verdict, or fix `PYTHONHASHSEED` and the timeout margin.
8. README, git history, Dockerfile.

### Optional

1. BugsInPy as a third held-out source.
2. Frozen strong-injector arm (Gemini as bug writer).
3. Five seeds for Tier 1.
4. Split A2 into separate diversity-bonus and pool ablations.
5. Coverage measurement; tests for the scripts.
6. Fix the `step` labelling quirk in `metrics.jsonl`.

---

## N. Strongest claims we can make (all supported by evidence)

1. A **test-verified training corpus of 25,000 Python functions** (24,048 train / 952 val) built from 484,097 KodCode rows. Every function passed its own unit tests twice in a sandbox. Data is de-duplicated (106,054 exact, 352 near-duplicates) and hash-recorded.
2. A **held-out benchmark of 3,500 real bug-fix pairs** from PyPIBugs and TSSB-3M, all reconstructed at function level, labelled with the same AST classifier as training, split dev/test without pair leakage, and checked for near-duplicates against the corpus (0 found). All 10 taxonomy classes have ≥143 real examples.
3. A **10-class bug taxonomy with a deterministic AST-diff classifier** and a format-preserving mutation engine. Every class round-trips in tests, and the classifier's labels agree with PyPIBugs' own `ArgSwap` labels (219/221 on the current held-out set).
4. A **six-check validity gate** whose behavioural check (G6) rejects equivalent mutants. About 12% of otherwise valid AST-engine single edits are behaviour-preserving under the tests (117/952; 362/3,000). G6 verdicts are 99.9% repeatable, and 98% of acceptances come from genuinely failing tests.
5. **Baselines established on real bugs:** the base Qwen2.5-Coder-1.5B localizes 1.9% of real dev bugs zero-shot (15.7% few-shot, but then it flags 99.9% of clean code). pyflakes localizes 9.1% with an 85.7% false-positive rate. These quantify the gap the RL detector must close.
6. **The injector cold-start problem was diagnosed and solved.** The base model produced 0/256 gate-valid mutants; after a 2,628-example format warmup (loss 0.59 → 0.11) the pass rate is 48.0%, with ≥1 valid mutant for 32/32 functions. That is the precondition for GRPO to have a learning signal.
7. **A complete GRPO co-training implementation:**
   - two LoRA adapters on one frozen base;
   - vLLM multi-LoRA hot-swapping;
   - token-level GRPO with k3 KL to the base model;
   - opponent pools, diversity tracker and curriculum;
   - checkpoint/resume.
   It has run end to end on CPU for every implemented ablation arm and end to end on the GPU (0.5B, 6 steps).
8. **126 automated tests, all passing**, including correctness tests of the log-probability computation against a full forward pass, adapter isolation, gradient direction, and checkpoint/resume.

---

## O. Claims we MUST NOT make yet

1. That adversarial co-training **improves** bug detection, or any comparison between arms. No co-training run exists.
2. Any RL detector accuracy, learning curve, or sim-to-real gap value. The gap metrics are implemented, not measured.
3. That GRPO "works" or "converges" here. Only the unit tests and the 0.5B smoke run exist. The smoke run never updated the injector, and its detector updates used clean inputs only.
4. Any A1–A7 finding (collapse, stability, curriculum effects). A4 and A5 are not even implemented.
5. That Gemini scores ~0.8 on real bugs. n = 30, all PyPIBugs.
6. Any Claude result (not run), SFT-baseline result (not implemented), or TRL-equivalence result (not run).
7. That the gate guarantees semantic non-equivalence. It guarantees a behaviour change *under the provided tests*, with slight run-to-run non-determinism.
8. That the 0.5B smoke run demonstrates learning. It demonstrates only that the pipeline runs.
9. Any 1.5B co-training step time or memory figure as measured. These are estimates.
10. That results are reproducible from a released artifact. There is no Docker image or git history yet.

---

## P. Recommended 50–75% progress narrative

The project is at roughly two-thirds completion: about 65%, with a defensible range of 58–72% depending on how heavily experimental results are weighted. The data and environment layers are essentially complete and validated.
- **Data:** a 25,000-function training corpus in which every function passes its own unit tests, a 3,500-pair real-bug benchmark drawn from two public bug-fix datasets, and a balanced synthetic validation set, all with recorded hashes.
- **Environment:** the 10-class taxonomy, the mutation engine, the AST classifier and the six-stage validity gate. The gate's key safeguard, the behavioural test that rejects equivalent mutants, was shown to reject roughly one in eight otherwise-valid edits that would otherwise have been unwinnable "bugs", with verdicts that are 99.9% repeatable.

The reinforcement-learning system is implemented and covered by a 126-test suite. It comprises GRPO with a KL penalty to the frozen base model, two LoRA policies sharing one base, vLLM-served rollouts with adapter hot-swapping, opponent pools, a diversity incentive, a curriculum and checkpointing. It has been exercised end to end on CPU for each implemented ablation arm and once on the GPU with a small model. Preliminary experiments established the baselines the method must beat: on real bugs, the base 1.5B model finds the correct line only 1.9% of the time, and a static linter 9.1%. They also exposed and solved a cold-start problem: the base model could not produce a single gate-valid bug (0 of 256), and a short supervised format warmup raised this to 48%. Without that fix, the adversarial game could not begin.

The remaining third is the experimental core: the co-training runs for ablations A1–A7 and their analysis. These need roughly 19–21 GB of GPU memory, against the ~6 GB currently free on a shared GPU. Two ablation mechanisms (PPO for A4, the group-size sweep for A5), the GRPO validation against a reference implementation, a supervised-detector baseline and the plotting code remain to be written. The project therefore has a validated foundation and a ready, tested training system, but no co-training results yet, and its research conclusions are still pending.

---

## Q. Exact evidence locations

| Result | File | Key / line / command |
|---|---|---|
| Corpus counts | `data/corpus/manifest.json` | `counts`, `stats.*`, `subsets`, `sha256`, `build_seconds` |
| Held-out counts | `data/heldout/manifest.json` | `pairs`, `pairs_by_split`, `classes`, `classes_by_source`, `locations`, `dropped`, `corpus_dedup_checked`, `sha256` |
| Synthetic-val | `data/synthetic/manifest.json` | `mutants`, `classes`, `locations`, `sha256` |
| Linter baseline | `results/linter/{dev,synthetic}.metrics.json` | `overall.*`, `source.*`, `class.*`, `location.*` |
| Base zero-shot | `results/base/{dev,synthetic}.metrics.json`; `logs/eval_base_splitdev.log`, `logs/eval_base_splitsynthetic.log` | `overall.*`; last log line |
| Base few-shot | `results/base-fewshot/dev.metrics.json`; `logs/eval_base_splitdevfewshot.log` | `overall.*` |
| Per-example predictions (for paired tests) | `results/*/<split>.predictions.jsonl` | `correct`, `line_exact`, `pred_class`, `pred_line` |
| Gemini samples | `results/gemini-*/dev.metrics.json`, `dev.gemini_cache.jsonl` | `overall.*`; `input_tokens`, `thinking_tokens`, `served_by`, `finish_reason` |
| Injector probe, base | `logs/probe_injector_base.log` | lines 1–4 |
| Injector probe, warmed | `logs/probe_injector_warm.log` | "256 samples: gate pass rate 0.480" |
| Warmup | `runs/warmup_injector/manifest.json`; `logs/warmup_injector.log` | `examples`, `classes`, `loss_first`, `loss_last`; "sft step N" lines |
| Warmed adapter | `runs/warmup_injector/injector/adapter_model.safetensors`, `adapter_config.json` | r = 32, `lora_alpha` = 64 |
| Gate funnel (AST engine) | `logs/audit_gate_funnel.json` | `synthetic_val.rejected`, `injector_warmup.rejected`, `replication` |
| Gate examples | `logs/audit_gate_examples.txt` | one block per example |
| G6 repeatability | `logs/audit_g6.json` | `run1_outcomes`, `verdict_agreement`, `flipped` |
| Smoke run | `runs/smoke_05b-s0/metrics.jsonl`, `config.json`, `checkpoint.pt`; `logs/smoke_05b.log` | `time/step_seconds`, `gate/*`, `detector/*`, `real_dev/*`; checkpoint key `step` = 6 |
| 0.5B raw outputs | `logs/probe_05b_raw_outputs.log` | "injector parseable 2/12, detector parseable 2/12" |
| GPU identity | `logs/gpu_info.txt` | `nvidia-smi --query-gpu=name,memory.total,driver_version` |
| Tests | command in §K | "126 passed in 31.28s" |
| Arm definitions | `configs/base.yaml`, `configs/arms/*.yaml` | — |
| Credentials | `.env` (git-ignored, mode 600) | a Gemini API key is configured; **value deliberately not reproduced** |

---

# Appendix 1: Component inventory

Columns: implemented / has tests / tests pass / experimentally validated.

| Component | File(s) | Main class/function | What it does | Impl. | Tests | Pass | Validated | Evidence | Limitations |
|---|---|---|---|---|---|---|---|---|---|
| Taxonomy | `taxonomy.py` | `BugClass`, `PRIORITY`, `DESCRIPTIONS`, `TAXONOMY_FINGERPRINT` | 10 disjoint classes, prompt text, version hash | ✅ | indirect (all classifier/mutator tests) | ✅ | ✅ used to label 3,500 real bugs | manifests carry `d6b0e82f9b10` | `loop bound` is a tag only; `resource leak` deferred |
| Mutation engine | `mutation/operators.py` | `mutate`, `mutate_any`, `_SiteCollector`, `_Replace` | Lists edit sites per class with libcst, applies one, keeps it only if `classify` agrees | ✅ | 42 | ✅ | ✅ 836 + 2,628 gated outputs | data manifests | `none_guard` sites rare in KodCode |
| AST classifier | `mutation/classify.py` | `diagnose`, `classify`, `_diffs`, `_RULES`, `_is_argument_swap`, `_location` | Structural AST diff → single-cause check → first matching class in priority order + location tag | ✅ | 31 | ✅ | ✅ labels real bugs (74 "other") | heldout manifest | `x or default` rule over-captures (Apps_13694_I) |
| Line utilities | `mutation/lines.py` | `changed_span`, `edited_line_count` | Bug lines in mutant coordinates; edit size | ✅ | in classify tests | ✅ | ✅ | — | — |
| Sandbox | `sandbox.py` | `run_tests`, `SandboxPool`, `RunResult` | Forked, rlimited test execution with escape routes blocked | ✅ | 6 | ✅ | ✅ 25k corpus, G6 audits | `audit_g6.json` | not a security boundary (no network namespace); flaky tests flip ~0.1% |
| Corpus | `data/corpus.py` | `build_corpus`, `static_reject_reason`, `split` | Filter, de-duplicate, verify tests, split | ✅ | none direct | — | ✅ | corpus manifest | — |
| Held-out | `data/heldout.py` | `to_examples`, `reverse_apply`, `read_pypibugs`, `read_tssb`, `assign_splits` | Fetch fixed file, reverse the patch, extract the function, label, split | ✅ | 5 | ✅ | ✅ | heldout manifest | Python 2 files dropped (1,106 TSSB) |
| Synthetic | `data/synthetic.py` | `uniform_mutants`, `to_examples` | Class-balanced gated AST mutants | ✅ | via loop tests | ✅ | ✅ | synthetic manifest | — |
| Prompts / parsers | `game/prompts.py` | `injector_messages`, `detector_messages`, `parse_injector`, `parse_detector`, `apply_action`, `format_injector_output` | Prompt construction, strict parsing, edit application | ✅ | 7 + 1 | ✅ | ✅ (format failure 0.1% for 1.5B detector) | `results/base` | strict format; 0.5B cannot comply |
| Gate | `game/gate.py` | `Gate.check`, `check_mutants`, `static_checks`, `_artifact`, `edit_chars` | G1–G6 | ✅ | 15 | ✅ | ✅ | §F | G6 is equivalence under tests |
| Detector reward | `game/rewards.py` | `detector_reward` | Binary or partial | ✅ | 1 (8 asserts) | ✅ | GPU smoke (clean inputs only) | `metrics.jsonl` | sparse |
| Injector reward | `game/rewards.py` | `injector_reward`, `fool_rate`, `subtlety` | Aggregate (§I) | ✅ | 1 | ✅ | CPU end-to-end only | test_loop_smoke | not exercised on GPU with valid mutants |
| Diversity tracker | `game/rewards.py` | `DiversityTracker` | EMA class frequency → bonus | ✅ | 1 | ✅ | CPU end-to-end | — | saturates at 1 for rare classes |
| LoRA policy | `rl/policy.py` | `Policy`, `completion_logprobs`, `use`, `save_adapter`, `load_adapter_weights`, `chat_prompt_ids` | Frozen base + named adapters; masked completion log-probs | ✅ | 6 | ✅ | ✅ SFT on GPU; 0.5B smoke | warmup manifest | — |
| vLLM rollout | `rl/rollout.py` | `VLLMGenerator` | Multi-LoRA sampling; new int id per adapter version | ✅ | none (GPU) | — | ✅ evals, probes, smoke | logs | a prompt over `max_model_len` hangs (no length guard) |
| HF rollout | `rl/rollout.py` | `HFGenerator` | CPU sampling for tests | ✅ | 1 | ✅ | n/a | — | ignores adapter versions |
| GRPO | `rl/grpo.py` | `group_advantages`, `policy_loss`, `zero_variance_fraction` | §H | ✅ | 4 | ✅ | GPU smoke only | — | no sampler/trainer importance correction |
| Co-training loop | `rl/loop.py` | `CoTrainer.step`, `_update`, `evaluate` | §G | ✅ | 5 | ✅ | GPU smoke (0.5B) | `runs/smoke_05b-s0` | no 1.5B run |
| Pools | `rl/loop.py` | `_snapshot_pools`, step l.132–163 | Past injector data, past detector scoring | ✅ | asserted counts | ✅ | GPU snapshots only | `checkpoint.pt` pools | — |
| Curriculum | `rl/loop.py` | step l.165–178 | A7 phases | ✅ | runs (no phase assertion) | ✅ | ❌ | — | definition awaiting confirmation |
| SFT warmup | `rl/sft.py`, `scripts/warmup_injector.py` | `sft` | Completion-only cross-entropy | ✅ | none direct | — | ✅ 0% → 48% gate pass | §E | biases injector start distribution |
| Metrics | `eval/metrics.py` | `score`, `is_correct`, `per_class_f1`, `expected_calibration_error` | All detector metrics | ✅ | 2 | ✅ | ✅ (except ECE) | `results/` | ECE not wired in |
| Baselines | `eval/baselines.py`, `scripts/eval.py` | `linter_predictions`, `claude_predictions`, `gemini_predictions`, `AnswerCache` | External/static detectors | ✅ | 2 | ✅ | partial | `results/` | Claude unrun; Gemini n = 30 |
| Statistics | `eval/stats.py` | `mcnemar`, `paired_bootstrap`, `hierarchical_bootstrap`, `holm` | Significance tests | ✅ | 2 | ✅ | ❌ (nothing to compare yet) | — | unused by scripts |
| Configuration | `config.py`, `configs/` | `load_config`, `Config` | YAML + arm override + dotted CLI overrides; rejects unknown keys | ✅ | indirect | ✅ | ✅ | `runs/*/config.json` | — |
| Checkpointing | `rl/loop.py` | `save_checkpoint`, `load_checkpoint` | Optimizer, scheduler, RNG, pools, adapter refs, diversity, curriculum EMA | ✅ | 1 | ✅ | ✅ written on GPU | `checkpoint.pt` | resume not run on GPU |
| Logging | `rl/loop.py` | `RunLog` | `config.json`, `metrics.jsonl`, `samples.jsonl` (+ W&B) | ✅ | indirect | ✅ | ✅ | smoke run | `step` labelling quirk; samples only for valid mutants |
| Scripts | `scripts/*.py` | 10 CLIs | Build/train/eval/probe/audit | ✅ | ❌ | — | ✅ all executed | `logs/` | untested |
| GPU smoke | `configs/arms/smoke_05b.yaml` | — | 0.5B 6-step run | ✅ | — | — | ✅ (pipeline only) | §J | 0.5B unusable as a policy |

# Appendix 2: Code architecture map

`scripts/train.py` builds the pieces in this order:
1. `load_config`;
2. read the corpus;
3. `SandboxPool` (spawned before CUDA initialises);
4. `Gate(GateConfig, sandbox)`;
5. `VLLMGenerator(base, max_lora_rank, gpu_memory_utilization, max_model_len, max_loras, enforce_eager)`;
6. `Policy.from_pretrained(base, ["injector", "detector"], LoraSettings)`;
7. `CoTrainer(cfg, policy, generator, gate, corpus)`, which loads the `init_adapter` warmup, then publishes v0 adapters;
8. the loop `step()`, then `evaluate()` every N steps, then `save_checkpoint()`;
9. `RunLog` writes JSONL.

**Dependencies** (→ means "uses"):
- `operators` → `classify`, `lines`, `taxonomy`.
- `gate` → `prompts` (parse/apply), `classify` (G5), `lines` (G2, bug lines), `sandbox` (G6).
- `rewards` → `GateResult`, `DetectorPrediction`.
- `loop` → `gate`, `rewards`, `prompts`, `synthetic` (A1-uniform), `grpo`, `policy`, `rollout`, `metrics` (evaluate).
- `heldout` → `classify` (labels); `synthetic` → `operators`, `gate`.
- `eval.py` → `metrics`, `baselines`, `rollout`, `policy.chat_prompt_ids`.

**Inputs and outputs by module:**
- **classify:** (original, mutant) source strings → `Diagnosis(bug_class, location)` or None.
- **operators:** (code, class, rng) → `Mutation(bug_class, mutant, lines, location)` or None.
- **gate:** (function, injector text) or (function, mutant, class) → `GateResult(ok, failed_check, mutant, claimed_class, bug_lines, location, edit_chars)`.
- **policy:** (prompts, completions, adapter) → (log-probs [B, C], mask [B, C]).
- **rollout:** (prompt ids, AdapterRef or list, n, temperature, max_tokens) → `list[list[Completion(text, token_ids)]]`.
- **grpo:** rewards [G, K] → advantages; (logp, A, mask, N, config, ref_logp) → (loss, stats).
- **loop.step:** () → (metrics dict, sample dicts).

# Appendix 3: Experiment registry

`n/r` = not recorded.

| # | Experiment | Model | Seed | Steps | Batch / K / K_D | Learning rate / β | Data, n | Runtime | GPU | Key results | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Corpus build | — | 0 | — | — | — | KodCode, 484,097 rows | 579 s | CPU | 25,000 kept | corpus manifest |
| 2 | Held-out build | — | 0 | — | — | — | 6,436 fix records | n/r | CPU | 3,500 pairs | heldout manifest |
| 3 | Synthetic build | — | 0 | — | — | — | 952 functions | n/r | CPU | 836 mutants | synthetic manifest |
| 4 | Linter baseline | pyflakes 4.0.2 | — | — | — | — | dev 1,386; synthetic 1,672 | n/r | CPU | §D | `results/linter` |
| 5 | Base zero-shot, dev | Qwen2.5-Coder-1.5B-Instruct | vLLM seed 0 (greedy) | — | n = 1, T = 0, 32 tokens | — | 1,386 | n/r | L4 (0.24 share, eager, max_model_len 2048) | §D | `results/base/dev.*` |
| 6 | Base few-shot, dev | same | same | — | same | — | 1,386 | n/r | L4 (0.24, 4096) | §D | `results/base-fewshot` |
| 7 | Base zero-shot, synthetic | same | same | — | same | — | 1,672 | n/r | L4 | §D | `results/base/synthetic.*` |
| 8–10 | Gemini samples | 3.1 Pro (default / `low`), 3.8 Flash | provider default | — | 1 answer each; max 16,384 output tokens | — | first 30 dev rows | ~7 min (Pro default; terminal output, not logged) | API | §D | `results/gemini-*` |
| 11 | Injector probe, base | 1.5B base | 0 | — | 32 × 8, T = 1.0, 128 tokens | — | 256 samples | n/r | L4 | pass 0.000 | `logs/probe_injector_base.log` |
| 12 | Injector SFT warmup | 1.5B + injector LoRA | 0 | 165 optimizer steps, 1 epoch | batch 16 (micro 4) | 1e-4 / — | 2,628 examples | n/r | L4 | loss 0.589 → 0.107 | warmup manifest and log |
| 13 | Injector probe, warmed | 1.5B + warm adapter | 0 | — | 32 × 8 | — | 256 | n/r | L4 | pass 0.480 | `logs/probe_injector_warm.log` |
| 14 | Smoke co-training | 0.5B | 0 | 6 | 4 / 4 / 4 (train 16 + 4) | 1e-5 / 0.01 | corpus | 12.5–20.5 s per step | L4 (0.12) | §J | `runs/smoke_05b-s0` |
| 15 | 0.5B raw-output probe | 0.5B | — | — | 6 × 2 per role | — | 6 val functions | n/r | L4 | 2/12 parseable per role | `logs/probe_05b_raw_outputs.log` |
| 16 | Gate funnel replay | — | 0 | — | — | — | 952 + 3,000 | n/r | CPU | §F.3 | `logs/audit_gate_funnel.json` |
| 17 | G6 repeatability | — | 0 | — | 2 runs | — | 952 | n/r | CPU | 99.89% agreement | `logs/audit_g6.json` |

Final reward / F1 / FPR / gate pass rate for any **co-training** experiment: **not recorded, because no such experiment exists** (apart from the smoke run in §J).

# Appendix 4: Representative code for the report

| Topic | File | Function | Purpose | Key logic | Why it matters |
|---|---|---|---|---|---|
| Taxonomy | `src/advbug/taxonomy.py` | `BugClass`, `PRIORITY`, `_fingerprint` | Defines the closed label set | `StrEnum` of 10 classes; priority = declaration order; SHA-256 fingerprint of version + descriptions | Every metric and ablation is defined over it; the fingerprint is logged with every artifact |
| Mutation | `src/advbug/mutation/operators.py` | `mutate`, `_try_sites`, `_SiteCollector.visit_*` | Applies one random class-specific edit | Collect sites per class in one libcst pass; shuffle; apply; accept only if `diagnose(...)` returns the target class | Labels are correct by construction; formatting is preserved, so there are no reformatting tells |
| AST classifier | `src/advbug/mutation/classify.py` | `diagnose`, `_diffs`, `_RULES`, `_is_argument_swap` | Labels any (original, mutant) pair | Structural diff yielding node / field / child / list differences; exactly one difference, or one swapped call-argument pair; rules tried in priority order | One labeller for the gate (G5), the synthetic data and the real held-out bugs |
| Gate | `src/advbug/game/gate.py` | `Gate.check`, `static_checks`, `_behaviour` | Rejects invalid or hackable injector outputs | G1 parse → G2 size → G3 compile → G4 artifacts → G5 label → G6 tests in the sandbox | Primary defence against reward hacking (equivalent mutants, giveaway comments) |
| Detector reward | `src/advbug/game/rewards.py` | `detector_reward` | Scores one detector answer | Exact line ∈ bug_lines and class match, or no_bug on clean code; optional partial credit | The verifiable reward that GRPO optimises |
| Injector reward | `src/advbug/game/rewards.py` + `rl/loop.py` l.154–178 | `injector_reward`, `fool_rate`, `DiversityTracker` | Scores one injector sample | −1 if invalid, else min(f', cap) + 0.1·b + 0.05·s | Encodes the adversarial objective and the anti-collapse and curriculum ablations |
| GRPO advantage | `src/advbug/rl/grpo.py` | `group_advantages` | Critic-free baseline | (r − group mean) / (population std + 1e-4) | The defining GRPO component; zero-variance groups are logged for A5 |
| GRPO loss | `src/advbug/rl/grpo.py` | `policy_loss` | Policy-gradient objective | Clipped surrogate (ratio 1 on-policy) + β·k3 KL; token-level sum / N | Exact objective; documents the on-policy and token-normalisation choices |
| Log-probabilities | `src/advbug/rl/policy.py` | `completion_logprobs` | Masked token log-probabilities | Left-pad, recomputed positions, `logits_to_keep`, float32 log-softmax, completion mask | Correct credit assignment; memory-efficient with a 151k vocabulary |
| Co-training step | `src/advbug/rl/loop.py` | `CoTrainer.step` | One full round of the game | Generate → gate → detector answers (shared for both players' rewards) → fool rates → rewards → two GRPO updates → publish/pools → dynamics | The heart of the method; one detector pass serves both players |
| Rollout | `src/advbug/rl/rollout.py` | `VLLMGenerator.generate`, `_request` | Fast sampling from any adapter version | `TokensPrompt` + per-prompt `LoRARequest`; a new int id per version | No weight syncing is needed because the base is frozen |
| Checkpointing | `src/advbug/rl/loop.py` | `save_checkpoint`, `load_checkpoint` | Resumable runs | Atomic `torch.save` of optimizer, scheduler, RNG, pools, adapter refs, diversity, curriculum EMA; reload adapter weights | Multi-day runs on a shared GPU |
| Evaluation | `src/advbug/eval/metrics.py` | `score`, `is_correct`, `per_class_f1` | Detector metrics | Line distance to any touched line; type accuracy on in-taxonomy bugs; FPR on clean code; macro-F1 over present labels | Shared by training-time evaluation and `scripts/eval.py` |

# Appendix 5: Recommended tables and figures

| # | Table / figure | Data source (file → fields) |
|---|---|---|
| 1 | Project completion matrix | §B of this document |
| 2 | Dataset statistics (corpus funnel, held-out composition, synthetic-val) | `data/*/manifest.json` → `stats`, `counts`, `pairs_by_split`, `classes_by_source`, `dropped` |
| 3 | Bug-class distribution, real vs synthetic (grouped bar) | `data/heldout/manifest.json:classes`, `data/synthetic/manifest.json:classes` |
| 4 | Baseline comparison | `results/*/<split>.metrics.json:overall` (`accuracy`, `line_exact`, `type_acc`, `macro_f1`, `false_positive_rate`, `format_failure`) |
| 5 | Base-model prediction bias (histogram of predicted lines; class counts) | `results/base*/dev.predictions.jsonl` → `pred_line`, `pred_class` |
| 6 | Injector readiness before/after warmup (G1–G6 stacked bars) | `logs/probe_injector_base.log`, `logs/probe_injector_warm.log` (the "failures:" line) |
| 7 | Warmup loss curve | `logs/warmup_injector.log` ("sft step N: loss X", every 10 steps) |
| 8 | G6 analysis: outcome types and failing-tests histogram | `logs/audit_g6.json` → `run1_outcomes`, `failing_tests_per_accepted_mutant` |
| 9 | Software test summary | §K table |
| 10 | RL workflow (textual) and GRPO/reward formulation | §G.2, §H, §I |
| 11 | Experimental roadmap (Tier 1–3, GPU requirement) | §M; `PROGRESS.md` §7 |

# Appendix 6: Findings from this audit

1. `expected_calibration_error` and all of `eval/stats.py` are implemented and tested but unused by any script.
2. The `metrics.jsonl` `step` field is the pre-update index (evaluation rows are off by one in labelling).
3. G6 is slightly non-deterministic. 1 of 952 verdicts flipped between runs, and the warmup replay accepted 2,635 vs 2,628 recorded. Timeouts count as behaviour change.
4. The classifier's `x or default` → `x` rule labels removal of a non-None `or` fallback as `none_guard_omission`.
5. A prompt longer than vLLM's `max_model_len` makes vLLM raise an error and then hang. Training has no prompt-length guard (corpus prompts are well under 4,096 tokens, so it hasn't occurred there).
6. Dropping zero-advantage sequences also removes their KL term (a design choice worth stating).
7. There is no importance correction between vLLM sampling and HF scoring.
8. Per-sample injector-probe outputs were not saved (only 3 printed per probe); `probe_injector.py` should persist them.
