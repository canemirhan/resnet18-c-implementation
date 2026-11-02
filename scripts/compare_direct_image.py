"""Compare direct C image preprocessing with torchvision-preprocessed PyTorch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pytorch.model import ResNet18


def read_logits(path: Path) -> torch.Tensor:
    with path.open("rb") as file:
        header = file.read(16)
        if header[:4] != b"RNLG" or header[4:8] != (1).to_bytes(4, "little"):
            raise ValueError(f"invalid logits file: {path}")
        count = int.from_bytes(header[8:12], "little")
        if count != 1000 or header[12:16] != b"\0\0\0\0":
            raise ValueError(f"invalid logits header: {path}")
        values = file.read()
    if len(values) != 1000 * 4:
        raise ValueError(f"invalid logits payload: {path}")
    return torch.frombuffer(bytearray(values), dtype=torch.float32).clone()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="compiled C executable")
    parser.add_argument("weights", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("classes", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/test_outputs/direct_c_logits.bin"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.executable),
        str(args.weights),
        str(args.image),
        str(args.classes),
        str(args.output),
    ]
    subprocess.run(command, check=True)

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = ResNet18()
    model.load_state_dict(weights.get_state_dict(progress=True))
    model.eval()
    image = weights.transforms()(Image.open(args.image).convert("RGB"))
    with torch.no_grad():
        reference = model(image.unsqueeze(0)).squeeze(0)
    direct = read_logits(args.output)
    difference = (reference - direct).abs()
    relative = difference / reference.abs().clamp_min(1e-8)
    reference_probabilities = reference.softmax(0)
    direct_probabilities = direct.softmax(0)
    reference_top5 = reference.topk(5).indices.tolist()
    direct_top5 = direct.topk(5).indices.tolist()
    reference_top10 = reference.topk(10).indices.tolist()
    direct_top10 = direct.topk(10).indices.tolist()

    print("\nDoğrudan C görüntü yolu ile torchvision/PyTorch karşılaştırması:")
    print(f"maksimum mutlak hata: {difference.max().item():.9g}")
    print(f"ortalama mutlak hata: {difference.mean().item():.9g}")
    print(f"maksimum bağıl hata: {relative.max().item():.9g}")
    print(f"top-1 aynı mı: {reference.argmax().item() == direct.argmax().item()}")
    print(f"top-5 aynı sırada mı: {reference_top5 == direct_top5}")
    print(f"top-10 aynı sırada mı: {reference_top10 == direct_top10}")
    print("PyTorch top-5:", [weights.meta["categories"][index] for index in reference_top5])
    print("C doğrudan görüntü top-5:", [weights.meta["categories"][index] for index in direct_top5])
    print("\nPyTorch ve C doğrudan görüntü top-10 olasılıkları:")
    print(f"{'Sıra':>4}  {'PyTorch sınıfı':<30} {'PyTorch':>10}  "
          f"{'C sınıfı':<30} {'C':>10}")
    for rank in range(10):
        reference_index = reference_top10[rank]
        direct_index = direct_top10[rank]
        print(
            f"{rank + 1:>4}  "
            f"{weights.meta['categories'][reference_index]:<30} "
            f"{reference_probabilities[reference_index].item():>9.4%}  "
            f"{weights.meta['categories'][direct_index]:<30} "
            f"{direct_probabilities[direct_index].item():>9.4%}"
        )


if __name__ == "__main__":
    main()
