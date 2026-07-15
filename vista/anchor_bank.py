from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import torch


def _normalize(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.detach().float().cpu().flatten()
    if scores.numel() == 0:
        raise ValueError("Anchor scores must not be empty")
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    scores = scores - scores.min()
    total = scores.sum()
    if total <= 0:
        return torch.full_like(scores, 1.0 / scores.numel())
    return scores / total


@dataclass
class AnchorBank:
    """Stateful visual-token selector used by VISTA drafting.

    Indices exposed by this class are ordinals within the visual-token list,
    not absolute KV positions. Runtime code performs the model-specific mapping.
    """

    initial_scores: torch.Tensor
    anchor_ratio: float = 0.1
    ema: float = 0.8
    attention_mass: float = 0.5
    scores: torch.Tensor = field(init=False)
    indices: torch.Tensor = field(init=False)
    refresh_count: int = field(default=0, init=False)
    jaccard_total: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if not 0 < self.anchor_ratio <= 1:
            raise ValueError("anchor_ratio must be in (0, 1]")
        if not 0 <= self.ema < 1:
            raise ValueError("ema must be in [0, 1)")
        if not 0 < self.attention_mass <= 1:
            raise ValueError("attention_mass must be in (0, 1]")
        self.scores = _normalize(self.initial_scores)
        self.indices = self._select(self.scores)

    @property
    def visual_token_count(self) -> int:
        return self.scores.numel()

    @property
    def budget(self) -> int:
        return max(1, min(self.visual_token_count, math.ceil(
            self.visual_token_count * self.anchor_ratio
        )))

    def _select(self, scores: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(scores, descending=True)
        cumulative = torch.cumsum(scores[order], dim=0)
        mass_count = int(
            torch.searchsorted(
                cumulative,
                torch.tensor(self.attention_mass, dtype=cumulative.dtype),
            ).item()
        ) + 1
        attention_count = min(self.budget, mass_count)
        selected = order[:attention_count].tolist()

        remaining_budget = self.budget - attention_count
        if remaining_budget:
            available = [
                idx for idx in range(self.visual_token_count) if idx not in set(selected)
            ]
            if remaining_budget >= len(available):
                selected.extend(available)
            else:
                uniform = torch.linspace(
                    0, len(available) - 1, remaining_budget
                ).round().long()
                selected.extend(available[idx] for idx in uniform.tolist())

        return torch.tensor(sorted(set(selected)), dtype=torch.long)

    def update(self, verify_scores: torch.Tensor) -> float:
        normalized = _normalize(verify_scores)
        if normalized.numel() != self.visual_token_count:
            raise ValueError(
                "Verifier score count does not match the visual-token count: "
                f"{normalized.numel()} != {self.visual_token_count}"
            )
        previous = self.indices
        self.scores = self.ema * self.scores + (1 - self.ema) * normalized
        self.scores = _normalize(self.scores)
        self.indices = self._select(self.scores)

        old_set = set(previous.tolist())
        new_set = set(self.indices.tolist())
        union = old_set | new_set
        jaccard = len(old_set & new_set) / len(union) if union else 1.0
        self.refresh_count += 1
        self.jaccard_total += jaccard
        return jaccard

    def metrics(self) -> Dict[str, float]:
        return {
            "visual_token_count": self.visual_token_count,
            "anchor_budget": self.budget,
            "anchor_refresh_count": self.refresh_count,
            "anchor_mean_jaccard": (
                self.jaccard_total / self.refresh_count
                if self.refresh_count else 1.0
            ),
        }
