"""One frozen base model carrying named LoRA adapters (injector, detector), plus log-probs.

The base never changes, so the KL reference is simply the base with adapters disabled, and an
adapter checkpoint (a few tens of MB) is all a generator or an opponent pool needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class LoraSettings:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.0
    target_modules: str | tuple[str, ...] = "all-linear"


def chat_prompt_ids(tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]]) -> list[int]:
    """Token ids of a chat prompt ending with the assistant header (shared by training and eval)."""
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


class Policy:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        adapters: Sequence[str],
        lora: LoraSettings,
        gradient_checkpointing: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.lora = lora
        config = peft.LoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=lora.target_modules if isinstance(lora.target_modules, str) else list(lora.target_modules),
            task_type="CAUSAL_LM",
        )
        self.model = peft.get_peft_model(model, config, adapter_name=adapters[0])
        for name in adapters[1:]:
            self.model.add_adapter(name, config)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self.adapters = tuple(adapters)
        self.use(adapters[0])

    @classmethod
    def from_pretrained(cls, base_model: str, adapters: Sequence[str], lora: LoraSettings, device: str = "cuda") -> Policy:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16).to(device)
        return cls(model, tokenizer, adapters, lora)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def adapter_parameters(self, name: str) -> list[torch.nn.Parameter]:
        return [p for n, p in self.model.named_parameters() if "lora_" in n and f".{name}." in n]

    def use(self, name: str) -> None:
        """Activate adapter `name`; only its parameters require gradients."""
        self.model.set_adapter(name)
        for n, p in self.model.named_parameters():
            p.requires_grad = "lora_" in n and f".{name}." in n

    def prompt_ids(self, messages: list[dict[str, str]]) -> list[int]:
        return chat_prompt_ids(self.tokenizer, messages)

    def completion_logprobs(
        self, prompts: Sequence[Sequence[int]], completions: Sequence[Sequence[int]], adapter: str | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token log-probs of each completion given its prompt: ([batch, C], mask [batch, C]).

        Sequences are left-padded so every completion ends at the last position; logits are
        computed only for the final C+1 positions (the 150k-token vocabulary makes full logits
        the largest memory cost). adapter=None scores with the frozen base (KL reference).
        """
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        c_max = max(len(c) for c in completions)
        seqs = [list(p) + list(c) for p, c in zip(prompts, completions)]
        length = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), length), pad, dtype=torch.long)
        attention = torch.zeros((len(seqs), length), dtype=torch.long)
        mask = torch.zeros((len(seqs), c_max), dtype=torch.bool)
        for i, (s, c) in enumerate(zip(seqs, completions)):
            input_ids[i, length - len(s) :] = torch.tensor(s)
            attention[i, length - len(s) :] = 1
            mask[i, c_max - len(c) :] = True
        input_ids, attention, mask = input_ids.to(self.device), attention.to(self.device), mask.to(self.device)
        positions = (attention.cumsum(-1) - 1).clamp(min=0)  # positions restart at 0 after left padding
        if adapter is None:
            ctx = self.model.disable_adapter()
        else:
            self.use(adapter)
            ctx = nullcontext()
        with ctx:
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention,
                position_ids=positions,
                logits_to_keep=c_max + 1,
                use_cache=False,
            ).logits[:, :-1].float()
        targets = input_ids[:, -c_max:]
        logp = torch.log_softmax(logits, dim=-1).gather(-1, targets[..., None]).squeeze(-1)
        return logp * mask, mask

    def save_adapter(self, name: str, directory: str | Path) -> Path:
        """Write adapter `name` so vLLM can load it from the returned directory."""
        directory = Path(directory)
        self.model.save_pretrained(directory, selected_adapters=[name])
        path = directory / name if (directory / name / "adapter_config.json").exists() else directory
        return path

    def load_adapter_weights(self, name: str, path: str | Path) -> None:
        state = peft.utils.load_peft_weights(str(path), device=str(self.device))
        peft.set_peft_model_state_dict(self.model, state, adapter_name=name)

