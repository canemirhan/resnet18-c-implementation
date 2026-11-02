"""Generate PyTorch logits for a normalized CHW input saved as a .pt tensor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pytorch.model import ResNet18


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PyTorch file containing a 3x224x224 tensor")
    parser.add_argument("output", type=Path, help="destination for logits as a PyTorch file")
    args = parser.parse_args()

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = ResNet18()
    model.load_state_dict(weights.get_state_dict(progress=True))
    model.eval()
    image = torch.load(args.input, map_location="cpu", weights_only=True)
    if image.shape != (3, 224, 224):
        raise ValueError(f"expected input shape (3, 224, 224), got {tuple(image.shape)}")
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(logits.squeeze(0), args.output)


if __name__ == "__main__":
    main()
