from __future__ import annotations

K_FACTOR: float = 32.0  # constant; tune later

__all__ = ["expected", "update", "K_FACTOR"]

expected = lambda r_p, r_q: 1.0 / (1.0 + 10 ** ((r_q - r_p) / 400.0))

def update(r_p: float, r_q: float, outcome: int) -> tuple[float, float]:
    """Single-step Elo update.

    outcome = 1 if player p wins else 0
    Returns (new_r_p, new_r_q).
    """
    delta = K_FACTOR * (outcome - expected(r_p, r_q))
    return r_p + delta, r_q - delta 