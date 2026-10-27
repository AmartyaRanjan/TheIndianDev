"""RL core on CPU with a tiny randomly initialised Qwen2 and the real Qwen tokenizer."""

import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from advbug.rl.grpo import LossConfig, group_advantages, policy_loss, zero_variance_fraction
from advbug.rl.policy import LoraSettings, Policy
from advbug.rl.rollout import AdapterRef, HFGenerator

BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def tiny_policy(tokenizer, seed=0, gradient_checkpointing=False):
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(config)
    return Policy(model, tokenizer, ["injector", "detector"], LoraSettings(r=4, alpha=8), gradient_checkpointing)


def _nudge(policy, name):
    """Make adapter `name` non-trivial (LoRA B starts at zero, i.e. identical to the base)."""
    with torch.no_grad():
        for p in policy.adapter_parameters(name):
            p.add_(0.05 * torch.randn_like(p))


def test_group_advantages_and_zero_variance():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    adv = group_advantages(rewards)
    assert torch.allclose(adv[0], torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-3)
    assert torch.all(adv[1] == 0)
    assert zero_variance_fraction(rewards) == 0.5


def test_policy_loss_on_policy_value_and_normalisation():
    logp = torch.log(torch.tensor([[0.5, 0.5], [0.25, 1.0]]))
    mask = torch.tensor([[True, True], [True, False]])
    loss, stats = policy_loss(logp, torch.tensor([1.0, -2.0]), mask, total_tokens=3, config=LossConfig(kl_beta=0.0))
    # on-policy: ratio == 1, so loss = -sum(A * mask) / total_tokens
    assert math.isclose(loss.item(), -(1 + 1 - 2) / 3, abs_tol=1e-6)
    assert stats["clip_fraction"] == 0.0
    _, stats = policy_loss(logp, torch.zeros(2), mask, 3, LossConfig(kl_beta=0.1), ref_logp=logp + 0.1)
    assert stats["kl"] > 0


def test_logprobs_independent_of_padding(tokenizer):
    policy = tiny_policy(tokenizer)
    _nudge(policy, "detector")
    prompts = [[1, 2, 3, 4, 5, 6], [7, 8]]
    completions = [[9, 10], [11, 12, 13, 14]]
    with torch.no_grad():
        batch, mask = policy.completion_logprobs(prompts, completions, "detector")
        for i in range(2):
            alone, _ = policy.completion_logprobs([prompts[i]], [completions[i]], "detector")
            n = len(completions[i])
            assert torch.allclose(batch[i][mask[i]], alone[0][-n:], atol=1e-5)
    assert mask.sum().item() == 6


def test_logprobs_match_full_forward(tokenizer):
    policy = tiny_policy(tokenizer)
    _nudge(policy, "detector")
    prompt, completion = [1, 2, 3, 4], [50, 51, 52]
    with torch.no_grad():
        scored, _ = policy.completion_logprobs([prompt], [completion], "detector")
        logits = policy.model(input_ids=torch.tensor([prompt + completion])).logits[0].float()
    full = torch.log_softmax(logits, -1)
    direct = torch.stack([full[len(prompt) - 1 + i, t] for i, t in enumerate(completion)])
    assert torch.allclose(scored[0], direct, atol=1e-5)


def test_adapters_are_isolated(tokenizer):
    policy = tiny_policy(tokenizer)
    prompts, completions = [[1, 2, 3]], [[4, 5]]
    with torch.no_grad():
        base0, _ = policy.completion_logprobs(prompts, completions, None)
        det0, _ = policy.completion_logprobs(prompts, completions, "detector")
    assert torch.allclose(base0, det0)  # zero-initialised LoRA B
    policy.use("injector")
    assert all(p.requires_grad for p in policy.adapter_parameters("injector"))
    assert not any(p.requires_grad for p in policy.adapter_parameters("detector"))
    _nudge(policy, "injector")
    with torch.no_grad():
        inj, _ = policy.completion_logprobs(prompts, completions, "injector")
        det1, _ = policy.completion_logprobs(prompts, completions, "detector")
        base1, _ = policy.completion_logprobs(prompts, completions, None)
    assert not torch.allclose(inj, base1)
    assert torch.allclose(det1, det0) and torch.allclose(base1, base0)


def test_one_gradient_step_moves_probability_the_right_way(tokenizer):
    policy = tiny_policy(tokenizer, gradient_checkpointing=True)
    _nudge(policy, "detector")
    params = policy.adapter_parameters("detector")
    opt = torch.optim.SGD(params, lr=0.05)  # small enough to stay in the linear regime
    prompts, completions = [[1, 2, 3], [1, 2, 3]], [[50, 51], [60, 61]]
    advantages = torch.tensor([1.0, -1.0])
    with torch.no_grad():
        before, mask = policy.completion_logprobs(prompts, completions, "detector")
    policy.model.train()
    logp, mask = policy.completion_logprobs(prompts, completions, "detector")
    loss, _ = policy_loss(logp, advantages, mask, int(mask.sum()), LossConfig(kl_beta=0.0))
    loss.backward()
    opt.step()
    policy.model.eval()
    with torch.no_grad():
        after, _ = policy.completion_logprobs(prompts, completions, "detector")
    gain = (after - before).sum(dim=1)
    assert gain[0] > 0 > gain[1]


def test_save_and_reload_adapter(tokenizer, tmp_path: Path):
    policy = tiny_policy(tokenizer)
    _nudge(policy, "detector")
    path = policy.save_adapter("detector", tmp_path / "v1")
    assert (path / "adapter_config.json").exists() and (path / "adapter_model.safetensors").exists()
    other = tiny_policy(tokenizer)  # same seed -> same base weights
    other.load_adapter_weights("detector", path)
    with torch.no_grad():
        a, _ = policy.completion_logprobs([[1, 2]], [[3, 4]], "detector")
        b, _ = other.completion_logprobs([[1, 2]], [[3, 4]], "detector")
    assert torch.allclose(a, b, atol=1e-6)


def test_hf_generator_shapes(tokenizer):
    policy = tiny_policy(tokenizer)
    generator = HFGenerator(policy)
    prompt = policy.prompt_ids([{"role": "user", "content": "hi"}])
    groups = generator.generate([prompt, prompt], [AdapterRef("detector", 0, "x"), None], n=3, temperature=1.0, max_tokens=5)
    assert len(groups) == 2 and all(len(g) == 3 for g in groups)
    assert all(1 <= len(c.token_ids) <= 5 for g in groups for c in g)
