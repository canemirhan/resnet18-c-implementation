"""Apply the official torchvision ResNet-18 preprocessing and save an input tensor."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu().contiguous().float().numpy().astype("<f4", copy=False)
    with path.open("wb") as file:
        file.write(b"RNIN")
        file.write((1).to_bytes(4, "little"))
        file.write((values.size).to_bytes(4, "little"))
        file.write((0).to_bytes(4, "little"))
        file.write(values.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    tensor = transform(Image.open(args.image).convert("RGB"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_tensor(args.output, tensor)
    print(f"wrote torchvision tensor: {args.output}")


if __name__ == "__main__":
    main()
