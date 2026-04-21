"""Risk guards (pre-trade checks)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskLimits:
    max_position_weight: float = 0.10   # single name cap
    max_gross_exposure: float = 1.0
    max_daily_turnover: float = 2.0


class RiskGate:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def check_weights(self, target_weights: dict[str, float]) -> list[str]:
        problems: list[str] = []
        if target_weights:
            gross = sum(abs(w) for w in target_weights.values())
            if gross > self.limits.max_gross_exposure + 1e-9:
                problems.append(f"gross {gross:.3f} > {self.limits.max_gross_exposure}")
            for ts_code, w in target_weights.items():
                if abs(w) > self.limits.max_position_weight + 1e-9:
                    problems.append(f"{ts_code} weight {w:.3f} exceeds cap")
        return problems
