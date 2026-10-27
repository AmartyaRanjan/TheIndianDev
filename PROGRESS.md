# Progress Report: Adversarial Self-Play for Code Bug Detection

*Status as of 2026-10-06. Companion to `PLAN.md` (the design) and `2rlproj.pdf` (the research spec).*

## 1. Where things stand

All of the work that does not need a GPU is finished: data, mutation engine, validity gate, rewards, the GRPO training loop, evaluation, and baselines. Each piece has been tested on CPU. As much GPU work as fits in the ~6 GB the other services leave free has also been done:
- the GPU check;
- the base-model baselines;
- the injector format warmup;
- a smoke test of the loop.

**The co-training runs themselves have not started.** They need about 19–21 GB of GPU memory, and only ~6.3 GB is free.

| Phase (PLAN.md §7) | State |
|---|---|
| 0 Environment | ✅ Done; GPU check passed in the free ~6 GB |
| 1 Taxonomy + mutation engine | ✅ Done |
| 2 Data (corpus, held-out, synthetic) | ✅ Done |
| 3 Prompts, metrics, baselines | ✅ Mostly done: linter and base model measured, Gemini sampled. Full Gemini run waits on a cost decision; Claude baseline not run (no key) |
| 3b Injector format warmup | ✅ Done (added after the base model produced 0 valid bugs) |
| 4 GRPO core | 🟡 Code done and CPU-tested; **the comparison against TRL (the external reference GRPO implementation) is not yet built** |
| 5 Full loop (injector, gate, pools, logging) | 🟡 Code done; tested on CPU and in a 0.5B GPU smoke run; **no 1.5B run yet** |
| 6 Pilot (100 steps, Tier 1) | ⏳ Needs GPU |
| 7 Tier 1 → Tier 2 → Tier 3 | ⏳ Needs GPU; Tier 2 code (A4 PPO, A5 configs) not written |
| 8 Figures, statistics, release | ⏳ Statistics code done; figures script and Dockerfile not written |

## 2. Machine and environment

**Software:** Python 3.12 via `uv` (`.venv`, 8.4 GB), torch 2.13 (CUDA 13.0), vLLM 0.31, transformers 5.17, PEFT 0.21, libcst 1.9, google-genai 2.28, anthropic 1.11. All pinned in `uv.lock`. The system Python (3.14) is not used.

**Models downloaded** (in `~/.cache/huggingface`): Qwen2.5-Coder-1.5B-Instruct (the base model) and Qwen2.5-Coder-0.5B-Instruct (smoke tests only).

**GPU (NVIDIA L4, 23,034 MiB, shared)**, as of the latest check:

| Account | GPU memory | Processes |
|---|---|---|
| `nupriya` | ~10.5 GB | ollama `llama-server` (5.8 GB), indicf5 TTS on port 8795 (3.0 GB), `simple-s2s` (1.7 GB) |
| UID `10001` | ~5.7 GB | four `/opt/venv` Python workers (~1.4 GB each) |
| `rajarshi` | 0 (was 5.0 GB) | stopped their fish-speech and F5 sidecars on 2026-10-06 |
| **Free** | **~6.3 GB** | |

**Disk:** 53 GB free. `data/raw/` holds 4.3 GB of downloads; the TSSB zip (0.9 GB) can be deleted, since the files are already extracted. `runs/smoke_05b-s0` (605 MB) can be deleted.

**Secrets:** the Gemini AI Studio key is in `.env` (git-ignored, mode 600). It was pasted into a chat transcript, so rotate it after use.

**Git:** the repository is initialised; **nothing has been committed yet**.

## 3. What has been built

### 3.1 Code map

| Path | Purpose |
|---|---|
| `src/advbug/taxonomy.py` | Taxonomy v1.0: 10 bug classes, prompt descriptions, fingerprint `d6b0e82f9b10` logged by every run |
| `src/advbug/mutation/operators.py` | Format-preserving libcst mutators, one family of edit sites per class |
| `src/advbug/mutation/classify.py` | AST-diff classifier: (original, mutant) → class. Used by gate check G5 and to label real bugs |
| `src/advbug/mutation/lines.py` | Changed-line span and edit size |
| `src/advbug/sandbox.py` | Runs unit tests in forked, rlimited children; `SandboxPool` of spawn workers |
| `src/advbug/data/corpus.py` | KodCode → filtered, de-duplicated, test-verified training corpus |
| `src/advbug/data/heldout.py` | PyPIBugs + TSSB-3M fix commits → function-level buggy/fixed pairs |
| `src/advbug/data/synthetic.py` | Gated AST-engine mutants (A1-uniform data, synthetic-val) |
| `src/advbug/game/prompts.py` | Injector/detector prompts, strict parsers, edit application, answer formatter |
| `src/advbug/game/gate.py` | Validity gate G1–G6 (format, size, compile, artifacts, label, behaviour change) |
| `src/advbug/game/rewards.py` | Detector rewards (binary or partial), injector reward (fool rate + diversity + subtlety), diversity tracker |
| `src/advbug/rl/policy.py` | One frozen base model with `injector`/`detector` LoRA adapters; completion log-probs |
| `src/advbug/rl/rollout.py` | `VLLMGenerator` (multi-LoRA hot-swap) and `HFGenerator` (CPU tests) |
| `src/advbug/rl/grpo.py` | Group advantages, clipped loss, k3 KL penalty |
| `src/advbug/rl/loop.py` | `CoTrainer`: one co-training step, pools, curriculum, checkpoint/resume, JSONL logging |
| `src/advbug/rl/sft.py` | Supervised fine-tuning of one adapter (injector warmup) |
| `src/advbug/eval/metrics.py` | Exact/within-N line accuracy, type accuracy, per-class F1, false-positive rate, ECE |
| `src/advbug/eval/baselines.py` | pyflakes linter, Claude (sync + batch), Gemini; on-disk answer cache |
| `src/advbug/eval/stats.py` | McNemar, paired and hierarchical bootstrap, Holm correction |
| `src/advbug/config.py`, `configs/` | Typed config; `base.yaml` + one override file per arm |
| `scripts/` | `build_corpus`, `build_heldout`, `build_synthetic`, `warmup_injector`, `probe_injector`, `train`, `run_queue`, `eval` |
| `tests/` | 126 tests, all passing on CPU, including end-to-end loop runs with a tiny random model for every Tier 1/3 arm |

### 3.2 Data assets

| Asset | Contents | Notes |
|---|---|---|
| `data/corpus/{train,val}.jsonl` | **24,048 train / 952 val** KodCode functions (8–40 lines, stdlib only) | Each passed its own tests twice. 484k rows scanned; 106k exact duplicates and 352 near-duplicates removed; 442 failed their tests, 9 flaky. All 12 KodCode subsets represented |
| `data/heldout/{dev,test}.jsonl` | **3,500 real bug pairs: 693 dev / 2,807 test** (buggy + fixed function per pair) | 1,766 PyPIBugs + 1,734 TSSB-3M. Every class ≥143 examples; 74 "other" (outside the taxonomy, scored for localization only). 0 near-duplicates of the corpus. Pairs never straddle splits |
| `data/synthetic/val.jsonl` | **836 gated mutants** + their 836 clean originals | Class-balanced except `none_guard_omission` (21) |
| `runs/warmup_injector/injector` | Injector LoRA after the format warmup | 2,628 gated examples, 1 epoch, loss 0.59 → 0.11 |

Each data folder has a `manifest.json` with counts, sha256 hashes and the taxonomy fingerprint.

**Held-out classes (all 3,500 pairs):** variable_misuse 1,037 · negation 404 · argument_swap 367 · boundary_shift 343 · comparison_op 273 · wrong_return 246 · arithmetic_op 222 · type_conversion 199 · none_guard_omission 192 · boolean_op 143 · other 74.

## 4. Results so far

### 4.1 Detector baselines

The full rows are scored on all 1,386 real-dev examples (693 bugs + 693 clean fixed versions) or all 1,672 synthetic-val examples. The Gemini rows are scored on the first 30 real-dev rows only, so treat them as a rough indication.

| Detector | Split | Accuracy | Line exact | Type acc. | False-positive rate | Format fail | Macro-F1 |
|---|---|---|---|---|---|---|---|
| pyflakes linter | dev | 0.072 | 0.091 | 0.000 | 0.857 | 0.000 | 0.020 |
| pyflakes linter | synthetic | 0.488 | 0.029 | 0.000 | 0.024 | 0.000 | 0.062 |
| Qwen 1.5B zero-shot | dev | 0.370 | 0.019 | 0.020 | 0.268 | 0.001 | 0.073 |
| Qwen 1.5B few-shot | dev | 0.013 | 0.157 | 0.119 | 0.999 | 0.000 | 0.055 |
| Qwen 1.5B zero-shot | synthetic | 0.465 | 0.016 | 0.011 | 0.074 | 0.000 | 0.071 |
| Gemini 3.1 Pro (default) | dev (30) | 0.667 | 0.800 | 0.800 | 0.467 | 0.033 | 0.816 |
| Gemini 3.1 Pro (`low` thinking) | dev (30) | 0.800 | 0.667 | 0.667 | 0.067 | 0.000 | 0.670 |
| Gemini 3.8 Flash | dev (30) | 0.833 | 0.867 | 0.867 | 0.200 | 0.000 | 0.926 |

*Accuracy* counts a bug as correct only with the exact line and the right type, and counts a clean function as correct only when the answer is "no bug".

**What these show:**
- **The base model has essentially no localization skill.** Zero-shot it answers "no bug" 74% of the time; few-shot it flags every function. In both modes its most common answer is line 10 (162 and 294 times). That is a positional habit, and the RL detector must beat it.
- **The linter flags 86% of clean real functions.** They are cut out of their modules, so names look undefined. As the plan expected, static checks don't find these semantic bugs.
- **Gemini is far stronger, and its thinking cost varies a lot.** Pro at default thinking used ~5,500 thinking tokens per request (max 15,729; one answer was cut off at the 16,384-token cap). Pro `low` used ~470 and Flash ~1,600.

### 4.2 Injector readiness

Same probe for each: 32 val functions × 8 samples, through the full gate.

| Injector | Gate pass | G1 format | G2 size | G3 compile | G4 artifact | G5 label | G6 equivalent | Functions with ≥1 valid bug |
|---|---|---|---|---|---|---|---|---|
| Base Qwen 1.5B | **0.000** | 0.574 | 0.164 | 0.160 | 0.004 | 0.098 | 0.000 | 0/32 |
| After format warmup | **0.480** | 0.031 | 0.098 | 0.039 | 0.000 | 0.324 | 0.027 | **32/32** |

Without the warmup every injector sample scores −1, the GRPO groups have no reward variance, and the injector never learns. After the warmup every group has a learning signal. The main remaining failure is claiming the wrong class (G5, 32%), which RL should reduce. `none_guard_omission` was not produced in the probe.

### 4.3 Training-loop smoke test (Qwen2.5-Coder-0.5B on GPU)

The loop ran 6 steps end to end on the GPU, at 13–20 s per step at tiny batch sizes. It exercised vLLM adapter hot-swapping, GRPO updates, adapter publishing and cleanup, the gate, evaluation and checkpoints. **The 0.5B model itself can't follow the output format**: 100% format failures at step 0, and 0% gate pass. It is unusable as a policy, even for smoke runs. Future smoke runs should use the 1.5B model with the warmed injector.

## 5. Decisions and deviations from the PDF

| Item | Decision | Why |
|---|---|---|
| Taxonomy | "Loop bound" is a location tag rather than a class; "resource leak" is deferred to v2 | Loop bound overlaps off-by-one and variable misuse; unit tests can't verify a leak |
| A7 curriculum | Phase A caps the fool-rate reward at 0.5 until the detector reaches 0.5 accuracy; Phase B then treats trivially caught bugs as invalid | The PDF wording supports more than one reading; this one is concrete |
| Injector warmup | Supervised format warmup shared by every arm; detector not warmed up | Base model produced 0 valid bugs (the plan's ≥80% compliance rule); warming the detector would blur A1 |
| Closed models | Claude/Gemini are baselines only, never trained policies | GRPO needs access to the weights |
| Gemini key | AI Studio key; default baseline model `gemini-3.1-pro-preview`, choosable per run | Strongest available; the exact `model_version` is recorded per answer |
| Claude baseline | Code ready (`claude-opus-5-5`, server-side refusal fallback on, serving model recorded); not run | No Anthropic key provided |
| vLLM context | Raised from 2,048 to 4,096 tokens | A few-shot prompt on a long dev function was 2,125 tokens |
| Corpus | KodCode (synthetic, test-backed) rather than mined GitHub code | Unit tests make the equivalent-mutant check (G6) possible |

## 6. Problems hit and fixed

- **Test harness:** some KodCode test files omit `from solution import ...`. The sandbox now pre-loads the solution's names, which took the candidate failure rate from ~33% to 0%.
- **Held-out fetching:** GitHub file paths with spaces crashed the build. They are now URL-quoted, and malformed URLs count as "unavailable".
- **`scripts/queue.py`** shadowed Python's standard `queue` module and broke vLLM's imports. It was renamed to `run_queue.py`.
- **Process hygiene:**
  - Killing a vLLM eval left its engine subprocess holding 5.2 GB of GPU memory.
  - The corpus build hung while shutting down and leaked 11 idle workers.
  - `SandboxPool.close()` now terminates its workers instead of waiting on them.
  - Lesson: kill vLLM jobs by PID, including the `VLLM::EngineCore` child, and never with `pkill -f` from a shell whose own command line contains the pattern.
- **transformers 5:** `apply_chat_template(tokenize=True)` now returns a dict. Prompts are rendered to text and then tokenized, in one shared function used by training and evaluation.

## 7. What is left: GPU runbook

Run these in order. Each step has an exit check; don't move on until it passes.

### Step 0: Get the GPU (prerequisite)

- **Needed:** ~19–21 GB free for the configuration in `configs/base.yaml` (vLLM share 0.35 ≈ 8 GB, plus ~8–10 GB for training and headroom).
- **Who to ask:** `nupriya` (~10.5 GB) and whoever runs the UID `10001` `/opt/venv` services (~5.7 GB).
- **If only ~17–18 GB comes free:** add `vllm.gpu_memory_utilization=0.25 vllm.enforce_eager=true optim.micro_batch=4` to every command below.
- **Check:** `nvidia-smi --query-gpu=memory.free --format=csv`.

### Step 1: 1.5B smoke run (~30 min)

```
uv run python scripts/train.py --arm configs/arms/full.yaml run_name=smoke15 steps=10 eval.every=5 \
    eval.real_dev_limit=100 eval.synthetic_limit=100
```

**Exit check:**
- no crash or out-of-memory error;
- gate pass rate in `runs/smoke15-s0/metrics.jsonl` around 0.4–0.5;
- `injector/trained_sequences` > 0 on most steps;
- note `time/step_seconds` for budgeting.

### Step 2: Validate the GRPO implementation against TRL (Phase 4 exit; **code not written yet**)

- **To write:** a script that trains the detector alone on A1-uniform data with TRL's `GRPOTrainer` and with our loop, using the same model, LoRA, data, group size and learning rate.
- **Exit check:** the two synthetic-accuracy curves agree within seed noise over ~200 steps.
- **Simpler fallback:** `run_queue.py --arms a1_uniform -- steps=200`. Synthetic and real-dev accuracy must clearly rise above the base model's (0.465 / 0.370).

### Step 3: Pilot, Tier 1 arms × 1 seed × 100 steps (~1 day)

```
uv run python scripts/run_queue.py --tier 1 --seeds 0 -- steps=100
```

**Exit check:**
- every arm runs and logs dynamics: class entropy, fool rate, sim-to-real gap, zero-variance groups;
- audit 50 mutants from `runs/full-s0/samples.jsonl` by hand: no equivalent mutants or giveaway artifacts.

**Then decide and freeze before Tier 1:**
- the final step count, from the measured step time (the plan assumes 400 steps at 1.5–2.5 min per step);
- hyperparameters (learning rate, KL β, λ_div, pool settings);
- commit the configs; that commit is the pre-registration.
- If the budget is too tight: KL β=0 (saves ~20% compute), 300 steps, or 2 seeds for the A5 control.

### Step 4: Tier 1, the main result (~7 days of GPU time at the plan's estimate)

```
uv run python scripts/run_queue.py --tier 1 --seeds 0 1 2
```

**Arms:** `full`, `a1_frozen`, `a1_uniform`, `a2_off`, `a3_off`, × 3 seeds = 15 runs. Runs resume after interruptions.

**Exit check:** all 15 reach the final step. Then score each final detector on **real-dev** and analyse:

```
uv run python scripts/eval.py --detector adapter --adapter runs/<arm>-s<seed>/adapters/detector/v<steps>/detector \
    --split dev --name <arm>-s<seed>
```

### Step 5: Tier 2 (**code still to write**, ~12 days)

- **A4 PPO:** a value head plus critic LoRA, GAE (γ=1, λ=0.95), the same rollouts and μ as GRPO (PLAN.md §4.6). Add `advantage: grpo|ppo` to the config.
- **A5 group-size sweep:** configs for K ∈ {2, 4, 16} with B·K = 128 held fixed, on `full`. Also the **stationary-opponent control**: the same sweep on `a1_uniform`, which the A5 question needs.

### Step 6: Tier 3 (~3 days)

`run_queue.py --tier 3 --seeds 0 1 2`. The `a6_partial` and `a7_curriculum` configs already exist.

### Step 7: Extra baseline (**code still to write**)

**Supervised detector on static mutations** ("why RL at all?"):
- fine-tune the detector adapter with `rl/sft.py` on AST-engine mutants of the training corpus, plus clean functions answered `no_bug`;
- match the RL arms' compute;
- score it like the arms.

### Step 8: Final evaluation and analysis

- **Real-test is touched only now.** Score every final checkpoint (all arms × seeds, plus baselines) on `--split test`.
- **Statistics** (`eval/stats.py`):
  - McNemar and paired bootstrap of each arm against `full`, Holm-corrected;
  - hierarchical bootstrap across seeds for confidence intervals.
- **Figures (`scripts/figures.py`, not written):** per arm, the injector class-entropy curve, the fool-rate curve, and the sim-to-real gap curve (`gap/*` in `metrics.jsonl`). Also per-class F1, a reliability diagram, and the A3 qualitative audit.
- **Release:** a Dockerfile (not written), the configs, the split hashes and the manifests.

## 8. Work that doesn't need the GPU

| Item | Blocked on |
|---|---|
| Gemini baseline on full real-dev (1,386 requests) | **Your choice of model/cost.** About 0.94M input tokens. Output: Pro default ~7.6M (≈ $90 at Gemini 3 Pro preview list prices), Pro `low` ~0.7M, Flash ~2.3M (much cheaper per token). Command: `uv run python scripts/eval.py --detector gemini --model gemini-3.8-flash --split dev --yes` |
| Gemini on real-test | Final evaluation only (Step 8) |
| Claude baseline | An Anthropic API key (or `ant auth login`) plus approval: ~$13–46 dev / $52–187 test at `claude-opus-5-5`, half with `--batch` |
| Optional arm: frozen strong injector (Gemini writes bugs, the gate filters, Qwen's detector trains) | Your go-ahead; check Gemini's terms on using outputs as training data |
| Kaggle dataset (from the PDF notes) | The link |
| First git commit | Your go-ahead |
| Rotate the Gemini key | Do after the Gemini runs |

## 9. Open questions

1. Which Gemini configuration should be the official strong-LLM baseline?
2. Should the frozen-strong-injector arm (Gemini as bug writer) be added?
3. Should the A2 diversity bonus and checkpoint pools be ablated separately, or together as now?
4. Should Tier 1 use 5 seeds instead of 3 if the GPU time allows?
5. Confirm the taxonomy deviations and the A7 definition (§5) before Tier 1 configs are frozen.

## Appendix: command reference

```
# tests (CPU)
uv run pytest -q

# rebuild data (CPU; already done)
uv run python scripts/build_corpus.py --target 25000
uv run python scripts/build_heldout.py
uv run python scripts/build_synthetic.py

# injector warmup and readiness probe (GPU; already done)
uv run python scripts/warmup_injector.py --functions 3000
uv run python scripts/probe_injector.py --adapter runs/warmup_injector/injector

# baselines
uv run python scripts/eval.py --detector linter --split dev
uv run python scripts/eval.py --detector base --split dev [--few-shot] [--eager --gpu-memory 0.24]
uv run python scripts/eval.py --detector gemini --model <model> --split dev [--thinking-level low] --yes
uv run python scripts/eval.py --detector claude --split dev [--batch] --yes

# training
uv run python scripts/train.py --arm configs/arms/<arm>.yaml [seed=N] [key=value ...] [--resume]
uv run python scripts/run_queue.py --tier 1 --seeds 0 1 2 [-- key=value ...]
```
