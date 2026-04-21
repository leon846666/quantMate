"""Factor registry — lets factors.py register functions with one decorator."""
from __future__ import annotations
from typing import Callable, Dict

FactorFn = Callable[..., object]

_FACTORS: Dict[str, FactorFn] = {}


def register_factor(name: str):
    def deco(fn: FactorFn) -> FactorFn:
        if name in _FACTORS:
            raise ValueError(f"Duplicate factor name: {name}")
        _FACTORS[name] = fn
        return fn
    return deco


def get_factor(name: str) -> FactorFn:
    if name not in _FACTORS:
        raise KeyError(f"Unknown factor: {name}. Registered: {list(_FACTORS)}")
    return _FACTORS[name]


def list_factors() -> list[str]:
    return sorted(_FACTORS.keys())


def all_factors() -> dict[str, FactorFn]:
    return dict(_FACTORS)
