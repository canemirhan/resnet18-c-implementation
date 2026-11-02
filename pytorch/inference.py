"""Run reproducible ImageNet inference with the official ResNet-18 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights

from .model import ResNet18


def write_tensor(path: Path, magic: bytes, tensor: torch.Tensor) -> None:
    values = tensor.detach().cpu().contiguous().float().numpy().astype("<f4", copy=False)
    with path.open("wb") as file:
        file.write(magic)
        file.write((1).to_bytes(4, "little"))
        file.write((values.size).to_bytes(4, "little"))
        file.write((0).to_bytes(4, "little"))
        file.write(values.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/test_outputs"))
    args = parser.parse_args()

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = ResNet18()
    model.load_state_dict(weights.get_state_dict(progress=True))
    model.eval()
    image = weights.transforms()(Image.open(args.image).convert("RGB"))
    with torch.no_grad():
        logits = model(image.unsqueeze(0)).squeeze(0)
    probabilities = logits.softmax(0)
    values, indices = probabilities.topk(10)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(image, args.output_dir / "input.pt")
    torch.save(logits, args.output_dir / "pytorch_logits.pt")
    write_tensor(args.output_dir / "input.bin", b"RNIN", image)
    write_tensor(args.output_dir / "pytorch_logits.bin", b"RNLG", logits)
    predictions = [
        {"index": int(index), "label": weights.meta["categories"][index], "score": float(value)}
        for value, index in zip(values, indices)
    ]
    (args.output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
    )
    for rank, prediction in enumerate(predictions, 1):
        print(f"{rank}. {prediction['label']}: {prediction['score']:.4%}")


if __name__ == "__main__":
    main()
