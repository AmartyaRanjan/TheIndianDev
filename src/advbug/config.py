"""Typed run configuration: configs/base.yaml, deep-merged with an arm file and `a.b=value` overrides."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class InjectorConfig:
    mode: str = "adversarial"  # adversarial | frozen | uniform  (A1)
    functions_per_step: int = 16  # B
    group_size: int = 8  # K
    temperature: float = 1.0
    max_tokens: int = 128
    init_adapter: str | None = None  # format-warmup adapter every arm starts from (scripts/warmup_injector.py)


@dataclass
class DetectorConfig:
    group_size: int = 4  # K_D
    train_mutants: int = 64
    train_clean: int = 16
    temperature: float = 1.0
    max_tokens: int = 32


@dataclass
class LoraConfig:
    r: int = 32
    alpha: int = 64


@dataclass
class OptimConfig:
    lr: float = 1e-5
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    micro_batch: int = 8
    kl_beta: float = 0.01
    clip_eps: float = 0.2
    skip_zero_advantage: bool = True  # samples with zero advantage contribute no policy gradient


@dataclass
class GateSection:
    enabled: bool = True  # A3


@dataclass
class RewardSection:
    partial_credit: bool = False  # A6
    invalid_penalty: float = -1.0
    lambda_div: float = 0.1  # A2 (with pool)
    lambda_sub: float = 0.05


@dataclass
class PoolConfig:
    enabled: bool = True  # A2
    every: int = 20
    keep: int = 10
    detector_data_fraction: float = 0.25  # share of detector mutant inputs from a past injector
    detector_weight: float = 0.3  # weight of a past detector in the injector's fool rate


@dataclass
class CurriculumConfig:
    enabled: bool = False  # A7
    threshold: float = 0.5
    cap: float = 0.5


@dataclass
class EvalConfig:
    every: int = 25
    real_dev_limit: int = 400
    synthetic_limit: int = 400


@dataclass
class VLLMConfig:
    gpu_memory_utilization: float = 0.35
    max_model_len: int = 4096
    max_loras: int = 6
    enforce_eager: bool = False


@dataclass
class Config:
    run_name: str = "full"
    seed: int = 0
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    backend: str = "vllm"  # vllm | hf (CPU tests)
    steps: int = 400
    corpus: str = "data/corpus/train.jsonl"
    synthetic_val: str = "data/synthetic/val.jsonl"
    real_dev: str = "data/heldout/dev.jsonl"
    out_dir: str = "runs"
    sandbox_workers: int = 4
    checkpoint_every: int = 25
    wandb: bool = False
    injector: InjectorConfig = field(default_factory=InjectorConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    gate: GateSection = field(default_factory=GateSection)
    rewards: RewardSection = field(default_factory=RewardSection)
    pool: PoolConfig = field(default_factory=PoolConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / f"{self.run_name}-s{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, data: dict[str, Any]) -> Any:
    kwargs = {}
    names = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"unknown config key {cls.__name__}.{key}")
        default = names[key].default_factory() if names[key].default_factory is not dataclasses.MISSING else None
        kwargs[key] = _build(type(default), value) if dataclasses.is_dataclass(default) else value
    return cls(**kwargs)


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _parse_override(item: str) -> dict[str, Any]:
    key, _, raw = item.partition("=")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    nested: dict[str, Any] = value
    for part in reversed(key.split(".")):
        nested = {part: nested}
    return nested


def load_config(base: str | Path, arm: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    data = yaml.safe_load(Path(base).read_text()) or {}
    if arm:
        data = _merge(data, yaml.safe_load(Path(arm).read_text()) or {})
    for item in overrides or []:
        data = _merge(data, _parse_override(item))
    return _build(Config, data)
