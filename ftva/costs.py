"""Cost model: API pay-per-token vs a self-hosted GPU you rent by the hour.

API prices are per million tokens, Claude API list prices checked Nov 2026
(Haiku 5.5 rates apply to prompts up to 100K tokens, which every note here is).
GPU prices vary a lot by provider: ALWAYS pass your real rate with --gpu-hourly.
"""
from __future__ import annotations

import math

API_PRICES = {  # USD per million tokens: (input, output)
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
BATCH_DISCOUNT = 0.5            # Claude Message Batches API: 50% off, results within 24h


def api_cost_per_doc(model: str, mean_in: float, mean_out: float, batch: bool = False) -> float:
    pin, pout = API_PRICES[model]
    c = (mean_in * pin + mean_out * pout) / 1e6
    return c * (BATCH_DISCOUNT if batch else 1.0)


def selfhost_cost_per_doc(gpu_hourly: float, docs_per_hour: float) -> float:
    """Marginal cost per document when the GPU is fully busy."""
    return gpu_hourly / docs_per_hour


def selfhost_cost_per_day(docs_per_day: float, gpu_hourly: float, docs_per_hour: float,
                          train_cost: float = 0.0, amortize_days: int = 30) -> float:
    """Always-on deployment: enough GPUs running 24h to carry the volume (at least one)."""
    gpus = max(1, math.ceil(docs_per_day / (docs_per_hour * 24)))
    return gpus * gpu_hourly * 24 + train_cost / amortize_days


def break_even_docs_per_day(api_cost: float, gpu_hourly: float, docs_per_hour: float,
                            train_cost: float = 0.0, amortize_days: int = 30) -> float | None:
    """Daily volume above which an always-on GPU becomes cheaper than the API.

    None means never: even a fully busy GPU costs more per document than the API.
    """
    if api_cost <= 0 or selfhost_cost_per_doc(gpu_hourly, docs_per_hour) >= api_cost:
        return None
    # walk up through GPU counts; within one count the self-host cost is flat
    capacity = docs_per_hour * 24
    for gpus in range(1, 10_000):
        fixed = gpus * gpu_hourly * 24 + train_cost / amortize_days
        v = fixed / api_cost
        if v <= gpus * capacity and v > (gpus - 1) * capacity:
            return v
        if v <= (gpus - 1) * capacity:          # cheaper as soon as this many GPUs are needed
            return (gpus - 1) * capacity
    return None
