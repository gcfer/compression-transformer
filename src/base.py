from __future__ import annotations

from typing import Protocol


class ProbabilityModel(Protocol):
    def symbol_limit(self) -> int:
        """Return the total number of symbols in the alphabet."""

    def total(self) -> int:
        """Return the total frequency mass."""

    def low(self, symbol: int) -> int:
        """Return the cumulative frequency below the symbol."""

    def high(self, symbol: int) -> int:
        """Return the cumulative frequency at or below the symbol."""

    def symbol_for_value(self, value: int) -> int:
        """Return the symbol whose cumulative interval contains value."""

    def update(self, symbol: int) -> None:
        """Update the model after encoding or decoding a symbol."""
