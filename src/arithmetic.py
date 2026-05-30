from __future__ import annotations

from base import ProbabilityModel


STATE_BITS = 32
FULL_RANGE = 1 << STATE_BITS
HALF_RANGE = FULL_RANGE >> 1
QUARTER_RANGE = HALF_RANGE >> 1
MIN_RANGE = QUARTER_RANGE + 2
MAX_TOTAL = MIN_RANGE
MASK = FULL_RANGE - 1


class BitOutputStream:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current_byte = 0
        self._num_bits_filled = 0

    def write(self, bit: int) -> None:
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        self._current_byte = (self._current_byte << 1) | bit
        self._num_bits_filled += 1
        if self._num_bits_filled == 8:
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._num_bits_filled = 0

    def finish(self) -> bytes:
        if self._num_bits_filled > 0:
            self._current_byte <<= 8 - self._num_bits_filled
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._num_bits_filled = 0
        return bytes(self._buffer)


class BitInputStream:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._position = 0
        self._current_byte = 0
        self._num_bits_remaining = 0

    def read(self) -> int:
        if self._num_bits_remaining == 0:
            if self._position >= len(self._data):
                return 0
            self._current_byte = self._data[self._position]
            self._position += 1
            self._num_bits_remaining = 8
        self._num_bits_remaining -= 1
        return (self._current_byte >> self._num_bits_remaining) & 1


class ArithmeticCoderBase:
    def __init__(self) -> None:
        self.low = 0
        self.high = MASK

    def update(self, model: ProbabilityModel, symbol: int) -> None:
        low_count = model.low(symbol)
        high_count = model.high(symbol)
        total = model.total()
        if total > MAX_TOTAL:
            raise ValueError("Frequency total too large for coder")

        current_range = self.high - self.low + 1
        new_low = self.low + low_count * current_range // total
        new_high = self.low + high_count * current_range // total - 1
        self.low = new_low
        self.high = new_high

        while ((self.low ^ self.high) & HALF_RANGE) == 0:
            self.shift()
            self.low = ((self.low << 1) & MASK)
            self.high = ((self.high << 1) & MASK) | 1

        while (self.low & ~self.high & QUARTER_RANGE) != 0:
            self.underflow()
            self.low = (self.low << 1) ^ HALF_RANGE
            self.high = ((self.high ^ HALF_RANGE) << 1) | HALF_RANGE | 1

    def shift(self) -> None:
        raise NotImplementedError()

    def underflow(self) -> None:
        raise NotImplementedError()


class ArithmeticEncoder(ArithmeticCoderBase):
    def __init__(self, bitout: BitOutputStream) -> None:
        super().__init__()
        self._bitout = bitout
        self._num_underflow = 0

    def write(self, model: ProbabilityModel, symbol: int) -> None:
        self.update(model, symbol)

    def finish(self) -> bytes:
        self._num_underflow += 1
        if self.low < QUARTER_RANGE:
            self._write_bit(0)
        else:
            self._write_bit(1)
        return self._bitout.finish()

    def shift(self) -> None:
        bit = self.low >> (STATE_BITS - 1)
        self._write_bit(bit)

    def underflow(self) -> None:
        self._num_underflow += 1

    def _write_bit(self, bit: int) -> None:
        self._bitout.write(bit)
        for _ in range(self._num_underflow):
            self._bitout.write(bit ^ 1)
        self._num_underflow = 0


class ArithmeticDecoder(ArithmeticCoderBase):
    def __init__(self, bitin: BitInputStream) -> None:
        super().__init__()
        self._bitin = bitin
        self.code = 0
        for _ in range(STATE_BITS):
            self.code = (self.code << 1) | self.read_code_bit()

    def read(self, model: ProbabilityModel) -> int:
        total = model.total()
        current_range = self.high - self.low + 1
        offset = self.code - self.low
        value = ((offset + 1) * total - 1) // current_range
        symbol = model.symbol_for_value(value)
        self.update(model, symbol)
        return symbol

    def shift(self) -> None:
        self.code = ((self.code << 1) & MASK) | self.read_code_bit()

    def underflow(self) -> None:
        self.code = (self.code & HALF_RANGE) | ((self.code << 1) & (MASK >> 1)) | self.read_code_bit()

    def read_code_bit(self) -> int:
        return self._bitin.read()
