from __future__ import annotations

from base import ProbabilityModel


class StaticFrequencyModel(ProbabilityModel):
    """Immutable cumulative-frequency model built from integer counts."""

    def __init__(self, counts: list[int]) -> None:
        if not counts:
            raise ValueError("counts must be non-empty")
        if any(count <= 0 for count in counts):
            raise ValueError("all counts must be positive")
        self._counts = counts
        cumulative = [0]
        running = 0
        for count in counts:
            running += count
            cumulative.append(running)
        self._cumulative = cumulative

    def symbol_limit(self) -> int:
        return len(self._counts)

    def total(self) -> int:
        return self._cumulative[-1]

    def low(self, symbol: int) -> int:
        return self._cumulative[symbol]

    def high(self, symbol: int) -> int:
        return self._cumulative[symbol + 1]

    def symbol_for_value(self, value: int) -> int:
        lo = 0
        hi = len(self._counts)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._cumulative[mid] > value:
                hi = mid
            else:
                lo = mid
        return lo

    def update(self, symbol: int) -> None:
        return None


def probs_to_counts(probabilities: list[float], scale: int = 1 << 14) -> list[int]:
    counts = [max(1, int(round(prob * scale))) for prob in probabilities]
    return counts
