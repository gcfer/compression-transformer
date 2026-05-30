from __future__ import annotations

import argparse
import io
import math
import re
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from codex_compression.entropy import (  # noqa: E402
    ArithmeticDecoder,
    ArithmeticEncoder,
    BitInputStream,
    BitOutputStream,
)
from codex_compression.models import StaticFrequencyModel, probs_to_counts  # noqa: E402


class ByteTransformer(nn.Module):
    def __init__(
        self,
        *,
        symbol_count: int,
        bos_symbol: int,
        context: int,
        width: int,
        layers: int,
        heads: int,
        hidden_mult: int,
    ) -> None:
        super().__init__()
        self.symbol_count = symbol_count
        self.bos_symbol = bos_symbol
        self.context = context
        self.token = nn.Embedding(symbol_count + 1, width)
        self.position = nn.Embedding(context, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * hidden_mult,
            dropout=0.0,
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Linear(width, symbol_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        hidden = self.token(x) + self.position(positions)
        return self.out(self.blocks(hidden))[:, -1, :]


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(path: str | None, length: int) -> bytes:
    if path:
        data = Path(path).read_bytes()
    else:
        corpus = Path("tests/corpora/divinecomedy.txt")
        data = corpus.read_bytes() if corpus.exists() else (b"to be or not to be\n" * 8000)
    return data[:length] if length else data


def build_text_tokenizer(data: bytes, vocab_size: int) -> tuple[list[bytes], dict[bytes, int], bytes]:
    """Build a tiny reversible tokenizer from frequent text chunks.

    Base symbols 0..255 still mean literal bytes. Extra symbols mean frequent
    byte chunks such as words or common punctuation/space runs. The dictionary
    is counted as side information after zlib compression.
    """
    if vocab_size <= 0:
        return [], {}, b""
    chunks = re.findall(rb"[A-Za-z]+|[0-9]+|[ \n\t\r]+|[^A-Za-z0-9 \n\t\r]+", data)
    counts = Counter(chunk for chunk in chunks if len(chunk) > 1)
    vocab = [chunk for chunk, _ in counts.most_common(vocab_size)]
    token_to_id = {chunk: 256 + i for i, chunk in enumerate(vocab)}
    raw_vocab = b"\n".join(len(chunk).to_bytes(2, "big") + chunk for chunk in vocab)
    return vocab, token_to_id, raw_vocab


def tokenize(data: bytes, token_to_id: dict[bytes, int]) -> list[int]:
    if not token_to_id:
        return list(data)
    symbols: list[int] = []
    chunks = re.findall(rb"[A-Za-z]+|[0-9]+|[ \n\t\r]+|[^A-Za-z0-9 \n\t\r]+", data)
    for chunk in chunks:
        token = token_to_id.get(chunk)
        if token is not None:
            symbols.append(token)
        else:
            symbols.extend(chunk)
    return symbols


def detokenize(symbols: list[int], vocab: list[bytes]) -> bytes:
    out = bytearray()
    for symbol in symbols:
        if symbol < 256:
            out.append(symbol)
        else:
            out.extend(vocab[symbol - 256])
    return bytes(out)


def prepare_symbols(
    data: bytes,
    tokenizer: str,
    text_vocab_size: int,
    zlib_level: int,
) -> tuple[list[int], list[bytes], int, int, int]:
    if tokenizer == "text":
        vocab, token_to_id, raw_vocab = build_text_tokenizer(data, text_vocab_size)
        symbols = tokenize(data, token_to_id)
        tokenizer_bytes = len(zlib.compress(raw_vocab, zlib_level))
    else:
        vocab = []
        symbols = list(data)
        tokenizer_bytes = 0
    eof_symbol = 256 + len(vocab)
    bos_symbol = eof_symbol + 1
    symbol_count = eof_symbol + 1
    return symbols + [eof_symbol], vocab, tokenizer_bytes, symbol_count, bos_symbol


def build_windows(symbols: list[int], context: int, bos_symbol: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for i in range(len(symbols)):
        row = [bos_symbol] + symbols[max(0, i - context + 1) : i]
        rows.append(([bos_symbol] * context + row)[-context:])
    return torch.tensor(rows, dtype=torch.long), torch.tensor(symbols, dtype=torch.long)


def ngram_distribution(
    history: list[int],
    counts1: list[int],
    counts2: dict[int, list[int]],
    counts3: dict[tuple[int, int], list[int]],
) -> list[float]:
    if len(history) >= 2 and tuple(history[-2:]) in counts3:
        counts = counts3[tuple(history[-2:])]
    elif history and history[-1] in counts2:
        counts = counts2[history[-1]]
    else:
        counts = counts1
    total = sum(counts)
    return [count / total for count in counts]


def update_ngram(
    history: list[int],
    symbol: int,
    counts1: list[int],
    counts2: dict[int, list[int]],
    counts3: dict[tuple[int, int], list[int]],
    symbol_count: int,
) -> None:
    counts1[symbol] += 1
    if history:
        counts2.setdefault(history[-1], [1] * symbol_count)[symbol] += 1
    if len(history) >= 2:
        counts3.setdefault(tuple(history[-2:]), [1] * symbol_count)[symbol] += 1


@torch.no_grad()
def transformer_probs_for_history(
    model: ByteTransformer,
    history: list[int],
    device: torch.device,
    cache: dict[tuple[int, ...], list[float]],
) -> list[float]:
    row = ([model.bos_symbol] * model.context + [model.bos_symbol] + history[-model.context + 1 :])[-model.context:]
    key = tuple(row)
    cached = cache.get(key)
    if cached is not None:
        return cached
    x = torch.tensor([row], dtype=torch.long, device=device)
    probs = torch.softmax(model(x)[0], dim=-1).cpu().tolist()
    cache[key] = probs
    return probs


def mixed_counts(
    t_probs: list[float],
    history: list[int],
    counts1: list[int],
    counts2: dict[int, list[int]],
    counts3: dict[tuple[int, int], list[int]],
    mix: float,
) -> list[int]:
    n_probs = ngram_distribution(history, counts1, counts2, counts3)
    probs = [(mix * tp) + ((1.0 - mix) * np) for tp, np in zip(t_probs, n_probs, strict=True)]
    return probs_to_counts(probs)


def estimate_payload_size(
    model: ByteTransformer,
    symbols: list[int],
    device: torch.device,
    mix: float,
) -> tuple[int, int, float]:
    """Estimate arithmetic payload size from ideal code length plus a small coder margin."""
    counts1 = [1] * model.symbol_count
    counts2: dict[int, list[int]] = {}
    counts3: dict[tuple[int, int], list[int]] = {}
    history: list[int] = []
    t_cache: dict[tuple[int, ...], list[float]] = {}
    ideal_bits = 0.0
    for symbol in symbols:
        t_probs = transformer_probs_for_history(model, history, device, t_cache)
        counts = mixed_counts(t_probs, history, counts1, counts2, counts3, mix)
        total = sum(counts)
        ideal_bits += -math.log2(counts[symbol] / total)
        update_ngram(history, symbol, counts1, counts2, counts3, model.symbol_count)
        history.append(symbol)
    low_bytes = math.ceil(ideal_bits / 8)
    high_bytes = low_bytes + 16 + math.ceil(len(symbols) * 0.002)
    return low_bytes, high_bytes, ideal_bits / max(1, len(symbols) - 1)


@torch.no_grad()
def coarse_estimate_payload_size(
    model: ByteTransformer,
    symbols: list[int],
    x: torch.Tensor,
    device: torch.device,
    mix: float,
    samples: int,
) -> tuple[int, int, float]:
    """Fast estimate from sampled positions using batched transformer calls.

    This does not simulate the adaptive n-gram model exactly at every byte. It
    uses prefix counts before each sampled position, then extrapolates.
    """
    usable = len(symbols)
    sample_count = min(samples, usable)
    if sample_count <= 0:
        return 0, 16, 0.0
    positions = torch.linspace(0, usable - 1, steps=sample_count).round().to(torch.long)
    positions = torch.unique(positions).cpu().tolist()
    batch = x[positions].to(device)
    t_probs_by_sample = torch.softmax(model(batch), dim=-1).cpu().tolist()

    counts1 = [1] * model.symbol_count
    counts2: dict[int, list[int]] = {}
    counts3: dict[tuple[int, int], list[int]] = {}
    history: list[int] = []
    next_sample = 0
    sample_bits: list[float] = []
    sampled_positions = set(positions)

    for position, symbol in enumerate(symbols):
        if position in sampled_positions:
            t_probs = t_probs_by_sample[next_sample]
            next_sample += 1
            counts = mixed_counts(t_probs, history, counts1, counts2, counts3, mix)
            total = sum(counts)
            sample_bits.append(-math.log2(counts[symbol] / total))
        update_ngram(history, symbol, counts1, counts2, counts3, model.symbol_count)
        history.append(symbol)

    mean_bits = sum(sample_bits) / len(sample_bits)
    ideal_bits = mean_bits * usable
    low_bytes = math.ceil(ideal_bits / 8)
    high_bytes = math.ceil((ideal_bits * 1.08) / 8) + 32
    return low_bytes, high_bytes, ideal_bits / max(1, len(symbols) - 1)


def encode(
    model: ByteTransformer,
    symbols: list[int],
    device: torch.device,
    mix: float,
) -> bytes:
    counts1 = [1] * model.symbol_count
    counts2: dict[int, list[int]] = {}
    counts3: dict[tuple[int, int], list[int]] = {}
    history: list[int] = []
    t_cache: dict[tuple[int, ...], list[float]] = {}
    bitout = BitOutputStream()
    coder = ArithmeticEncoder(bitout)
    for symbol in symbols:
        t_probs = transformer_probs_for_history(model, history, device, t_cache)
        counts = mixed_counts(t_probs, history, counts1, counts2, counts3, mix)
        coder.write(StaticFrequencyModel(counts), symbol)
        update_ngram(history, symbol, counts1, counts2, counts3, model.symbol_count)
        history.append(symbol)
    return coder.finish()


def decode(
    model: ByteTransformer,
    payload: bytes,
    device: torch.device,
    mix: float,
) -> list[int]:
    counts1 = [1] * model.symbol_count
    counts2: dict[int, list[int]] = {}
    counts3: dict[tuple[int, int], list[int]] = {}
    history: list[int] = []
    t_cache: dict[tuple[int, ...], list[float]] = {}
    bitin = BitInputStream(payload)
    coder = ArithmeticDecoder(bitin)
    while True:
        t_probs = transformer_probs_for_history(model, history, device, t_cache)
        counts = mixed_counts(t_probs, history, counts1, counts2, counts3, mix)
        symbol = coder.read(StaticFrequencyModel(counts))
        if symbol == model.symbol_count - 1:
            return history
        update_ngram(history, symbol, counts1, counts2, counts3, model.symbol_count)
        history.append(symbol)


def compressed_model_size(model: nn.Module, level: int) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(zlib.compress(buffer.getvalue(), level))


def quantize_tensor(tensor: torch.Tensor, bits: int) -> tuple[torch.Tensor, float, float]:
    if not tensor.is_floating_point():
        return tensor.detach().cpu(), 1.0, 0.0
    levels = (1 << bits) - 1
    data = tensor.detach().cpu()
    lo = float(data.min())
    hi = float(data.max())
    if hi == lo:
        return torch.zeros_like(data, dtype=torch.int32), 1.0, lo
    scale = (hi - lo) / levels
    quantized = torch.clamp(torch.round((data - lo) / scale), 0, levels).to(torch.int32)
    return quantized, scale, lo


def pack_bits(values: torch.Tensor, bits: int) -> bytes:
    packed = bytearray()
    current = 0
    filled = 0
    mask = (1 << bits) - 1
    for value in values.reshape(-1).tolist():
        current = (current << bits) | (int(value) & mask)
        filled += bits
        while filled >= 8:
            filled -= 8
            packed.append((current >> filled) & 0xFF)
    if filled:
        packed.append((current << (8 - filled)) & 0xFF)
    return bytes(packed)


def dequantize_tensor(quantized: torch.Tensor, scale: float, offset: float) -> torch.Tensor:
    if not quantized.is_floating_point():
        return quantized.to(torch.float32) * scale + offset
    return quantized


def quantize_state_dict(
    model: nn.Module,
    bits: int,
) -> tuple[dict[str, torch.Tensor], bytes]:
    if not 1 <= bits <= 16:
        raise ValueError("--quant-bits must be between 1 and 16")
    dequantized: dict[str, torch.Tensor] = {}
    quantized_package: dict[str, object] = {}
    for name, tensor in model.state_dict().items():
        q_tensor, scale, offset = quantize_tensor(tensor, bits)
        quantized_package[name] = {
            "shape": tuple(tensor.shape),
            "scale": scale,
            "offset": offset,
            "bits": bits,
            "data": pack_bits(q_tensor, bits),
        }
        dequantized[name] = dequantize_tensor(q_tensor, scale, offset).to(tensor.dtype)
    buffer = io.BytesIO()
    torch.save({"bits": bits, "state": quantized_package}, buffer)
    return dequantized, buffer.getvalue()


def apply_quantization(model: nn.Module, bits: int, zlib_level: int) -> tuple[int, int]:
    dequantized, raw_package = quantize_state_dict(model, bits)
    model.load_state_dict(dequantized)
    compressed_bytes = len(zlib.compress(raw_package, zlib_level))
    raw_bytes = len(raw_package)
    return raw_bytes, compressed_bytes


def elapsed(label: str, start: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    now = time.perf_counter()
    print(f"{label:<24} {now - start:8.3f}s")
    return now


def human_bytes(value: int | float) -> str:
    units = ["B", "KB", "MB", "GB"]
    number = float(value)
    for unit in units:
        if abs(number) < 1000 or unit == units[-1]:
            if unit == "B":
                return f"{number:.0f} B"
            return f"{number:.2f} {unit}"
        number /= 1000
    return f"{number:.2f} GB"


def human_range(low: int, high: int) -> str:
    if low == high:
        return human_bytes(low)
    return f"{human_bytes(low)} - {human_bytes(high)}"


def human_delta(value: int) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}{human_bytes(abs(value))}"


def parse_csv_numbers(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced transformer compressor experiment.")
    parser.add_argument("path", nargs="?")
    parser.add_argument("--length", type=int, default=0, help="compress only the first N bytes; 0 means all bytes")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hidden-mult", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--max-train-windows",
        type=int,
        default=0,
        help="train on at most this many sampled positions; 0 uses every position",
    )
    parser.add_argument("--torch-threads", type=int, default=0, help="PyTorch CPU threads; 0 leaves default")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--mix", type=float, default=0.75, help="transformer weight; rest is adaptive ngram")
    parser.add_argument("--tokenizer", choices=["byte", "text"], default="text")
    parser.add_argument("--text-vocab-size", type=int, default=512)
    parser.add_argument("--zlib-level", type=int, default=9)
    parser.add_argument(
        "--quant-bits",
        type=int,
        default=8,
        help="uniform post-training quantization bits per floating weight; 0 disables quantization",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--estimate-only", action="store_true", help="train and estimate size, but skip arithmetic coding")
    parser.add_argument(
        "--coarse-estimate-only",
        action="store_true",
        help="train and estimate from sampled positions; much faster but rougher",
    )
    parser.add_argument("--coarse-samples", type=int, default=1024, help="sample count for --coarse-estimate-only")
    parser.add_argument("--sweep-quant-bits", default="", help="comma list, e.g. 2,3,4,6,8")
    parser.add_argument("--sweep-mix", default="", help="comma list, e.g. 0.2,0.4,0.6,0.8")
    parser.add_argument("--skip-decode", action="store_true", help="run arithmetic encode but skip decode verification")
    args = parser.parse_args()

    if args.width % args.heads != 0:
        raise ValueError("--width must be divisible by --heads")
    if not 0 <= args.mix <= 1:
        raise ValueError("--mix must be between 0 and 1")

    device = choose_device(args.device)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    data = load_data(args.path, args.length)
    symbols, token_vocab, tokenizer_bytes, symbol_count, bos_symbol = prepare_symbols(
        data,
        args.tokenizer,
        args.text_vocab_size,
        args.zlib_level,
    )
    zip_bytes = len(zlib.compress(data, args.zlib_level))

    print(f"input size: {human_bytes(len(data))} ({len(data)} bytes)")
    print(f"zip baseline: {human_bytes(zip_bytes)} ({zip_bytes} bytes)")
    print(f"device: {device}")
    print(
        f"model: context={args.context}, width={args.width}, layers={args.layers}, "
        f"heads={args.heads}, mix={args.mix}"
    )
    print(
        f"tokenizer: {args.tokenizer}, symbols={symbol_count}, tokenized length={len(symbols)}, "
        f"zipped tokenizer={human_bytes(tokenizer_bytes)}"
    )

    t = time.perf_counter()
    torch.manual_seed(0)
    model = ByteTransformer(
        symbol_count=symbol_count,
        bos_symbol=bos_symbol,
        context=args.context,
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        hidden_mult=args.hidden_mult,
    ).to(device)
    x_cpu, y_cpu = build_windows(symbols, args.context, bos_symbol)
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    if args.max_train_windows and args.max_train_windows < len(x):
        train_positions = torch.linspace(0, len(x) - 1, steps=args.max_train_windows).round().long()
        train_positions = torch.unique(train_positions).to(device)
        train_x = x[train_positions]
        train_y = y[train_positions]
    else:
        train_x = x
        train_y = y
    print(f"training windows: {len(train_x)} of {len(x)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t = elapsed("setup", t, device)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_items = 0
        permutation = torch.randperm(len(train_x), device=device)
        for start in range(0, len(train_x), args.batch_size):
            idx = permutation[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            total_items += len(idx)
        print(f"epoch {epoch:>3}: loss={total_loss / total_items:.4f}")
    t = elapsed("training", t, device)

    model.eval()
    trained_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if args.sweep_quant_bits or args.sweep_mix:
        quant_values = parse_csv_numbers(args.sweep_quant_bits, int) if args.sweep_quant_bits else [args.quant_bits]
        mix_values = parse_csv_numbers(args.sweep_mix, float) if args.sweep_mix else [args.mix]
        results = []
        print("\nsweep")
        for quant_bits in quant_values:
            model.load_state_dict(trained_state)
            model.to(device)
            model.eval()
            if quant_bits:
                raw_bytes, zipped_model_bytes = apply_quantization(model, quant_bits, args.zlib_level)
                model.to(device)
                model.eval()
            else:
                raw_bytes = 0
                zipped_model_bytes = compressed_model_size(model, args.zlib_level)
            for mix in mix_values:
                if args.coarse_estimate_only:
                    low, high, bpb = coarse_estimate_payload_size(
                        model,
                        symbols,
                        x,
                        device,
                        mix,
                        args.coarse_samples,
                    )
                else:
                    low, high, bpb = estimate_payload_size(model, symbols, device, mix)
                total_low = low + zipped_model_bytes + tokenizer_bytes
                total_high = high + zipped_model_bytes + tokenizer_bytes
                results.append((total_low, total_high, quant_bits, mix, low, high, zipped_model_bytes, raw_bytes, bpb))
                print(
                    f"  q={quant_bits:>2} mix={mix:.2f} "
                    f"payload={human_range(low, high)} "
                    f"model={human_bytes(zipped_model_bytes)} "
                    f"total={human_range(total_low, total_high)} "
                    f"vs_zip={human_delta(total_low - zip_bytes)} to {human_delta(total_high - zip_bytes)}"
                )
        best = min(results, key=lambda row: row[0])
        print(
            "\nbest sweep result: "
            f"q={best[2]}, mix={best[3]:.2f}, total={human_range(best[0], best[1])}, "
            f"vs_zip={human_delta(best[0] - zip_bytes)} to {human_delta(best[1] - zip_bytes)}"
        )
        return

    if args.quant_bits:
        raw_model_bytes, model_bytes = apply_quantization(model, args.quant_bits, args.zlib_level)
        model.to(device)
        model.eval()
        print(
            f"quantization            {args.quant_bits} bits, "
            f"raw package {human_bytes(raw_model_bytes)}, zipped package {human_bytes(model_bytes)}"
        )
    else:
        raw_model_bytes = None
        model_bytes = compressed_model_size(model, args.zlib_level)
        print("quantization            disabled")
    if args.coarse_estimate_only:
        est_low, est_high, est_bpb = coarse_estimate_payload_size(
            model,
            symbols,
            x,
            device,
            args.mix,
            args.coarse_samples,
        )
        t = elapsed("coarse estimate", t, device)
    else:
        est_low, est_high, est_bpb = estimate_payload_size(model, symbols, device, args.mix)
        t = elapsed("payload estimate", t, device)

    payload: bytes | None = None
    if not args.estimate_only and not args.coarse_estimate_only:
        payload = encode(model, symbols, device, args.mix)
        t = elapsed("arithmetic encode", t, device)

        if not args.skip_decode:
            restored = decode(model, payload, device, args.mix)
            t = elapsed("arithmetic decode", t, device)
            assert detokenize(restored, token_vocab) == data
        else:
            print("arithmetic decode       skipped")

    actual_payload_bytes = len(payload) if payload is not None else None
    side_info_bytes = model_bytes + tokenizer_bytes
    total_est_low = est_low + side_info_bytes
    total_est_high = est_high + side_info_bytes
    total_actual = actual_payload_bytes + side_info_bytes if actual_payload_bytes is not None else None

    print("\nsize summary")
    print(f"original text:         {human_bytes(len(data)):>14} ({len(data)} bytes)")
    print(f"zip benchmark:         {human_bytes(zip_bytes):>14} ({zip_bytes} bytes)")
    if args.coarse_estimate_only:
        print(f"estimate mode:         coarse, {args.coarse_samples} samples")
    print(f"compressed file est.:  {human_range(est_low, est_high):>14}")
    if actual_payload_bytes is not None:
        print(f"compressed file actual:{human_bytes(actual_payload_bytes):>14} ({actual_payload_bytes} bytes)")
    else:
        print("compressed file actual: skipped")
    if raw_model_bytes is not None:
        print(f"quantized model raw:   {human_bytes(raw_model_bytes):>14} ({raw_model_bytes} bytes)")
    print(f"quantized model zipped:{human_bytes(model_bytes):>14} ({model_bytes} bytes)")
    print(f"tokenizer zipped:      {human_bytes(tokenizer_bytes):>14} ({tokenizer_bytes} bytes)")
    print(f"total est. file+model: {human_range(total_est_low, total_est_high):>14}")
    if total_actual is not None:
        print(f"total actual file+model:{human_bytes(total_actual):>13} ({total_actual} bytes)")
        print(f"total vs zip benchmark:{human_delta(total_actual - zip_bytes):>13}")
        print(f"payload bits/byte:     {8 * actual_payload_bytes / len(data):>8.3f}")
        print(f"with model bits/byte:  {8 * total_actual / len(data):>8.3f}")
    else:
        print(
            "total est. vs zip:     "
            f"{human_delta(total_est_low - zip_bytes)} to {human_delta(total_est_high - zip_bytes)}"
        )
        print(f"estimated bits/byte:   {est_bpb:>8.3f}")


if __name__ == "__main__":
    main()
