"""Compare C and PyTorch logits from the reproducible tensor artifacts."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch


def read_logits(path: Path) -> torch.Tensor:
    with path.open("rb") as file:
        header = file.read(16)
        magic, version, count, reserved = struct.unpack("<4sIII", header)
        if (magic, version, count, reserved) != (b"RNLG", 1, 1000, 0):
            raise ValueError(f"invalid logits file: {path}")
        values = file.read()
    if len(values) != 1000 * 4:
        raise ValueError(f"invalid logits payload: {path}")
    return torch.frombuffer(bytearray(values), dtype=torch.float32).clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("c_logits", type=Path)
    args = parser.parse_args()

    reference = read_logits(args.reference)
    c_logits = read_logits(args.c_logits)
    difference = (reference - c_logits).abs()
    denominator = reference.abs().clamp_min(1e-8)
    reference_top5 = reference.topk(5).indices.tolist()
    c_top5 = c_logits.topk(5).indices.tolist()
    reference_top10 = reference.topk(10).indices.tolist()
    c_top10 = c_logits.topk(10).indices.tolist()
    print(f"max absolute error: {difference.max().item():.9g}")
    print(f"mean absolute error: {difference.mean().item():.9g}")
    print(f"max relative error: {(difference / denominator).max().item():.9g}")
    print(f"top-1 agreement: {reference.argmax().item() == c_logits.argmax().item()}")
    print(f"top-5 agreement: {reference_top5 == c_top5}")
    print(f"top-10 agreement: {reference_top10 == c_top10}")


if __name__ == "__main__":
    main()
