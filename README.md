# Transformer Compression Experiment

This is a compact, reproducible experiment showing how a tiny Transformer can
compress text by predicting the next token, then arithmetic-coding the text from
those probabilities.

The reported size includes:

- the estimated arithmetic-coded payload
- the quantized model weights after zlib compression
- the text tokenizer vocabulary after zlib compression

The included corpus is `data/divinecomedy.txt`.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the Best Run

```bash
python compression_transformer.py data/divinecomedy.txt \
  --length 0 \
  --epochs 20 \
  --width 8 \
  --layers 1 \
  --heads 1 \
  --context 64 \
  --tokenizer text \
  --text-vocab-size 1024 \
  --quant-bits 4 \
  --mix 0.9 \
  --device cpu \
  --torch-threads 3 \
  --coarse-estimate-only \
  --coarse-samples 8192 \
  --max-train-windows 65536
```

On the machine used for this experiment, this produced approximately:

```text
zip baseline:          265.79 KB
compressed file est.:  227.93 KB - 246.20 KB
quantized model zipped: 12.02 KB
tokenizer zipped:        3.85 KB
total est. file+model: 243.81 KB - 262.07 KB
total est. vs zip:     -21.98 KB to -3.72 KB
```

The exact values can vary slightly across PyTorch versions and hardware.

## What It Does

1. Builds a simple reversible text tokenizer from frequent byte chunks.
2. Trains a small Transformer to predict the next token from recent context.
3. Quantizes the trained weights.
4. Estimates the arithmetic-coded payload size from model probabilities.
5. Adds the compressed model and tokenizer sizes to compare against zlib/ZIP.

This is an experiment, not a production compressor. The default run uses
`--coarse-estimate-only`, which is fast but approximate. The script also contains
full arithmetic encode/decode code paths for smaller runs.
