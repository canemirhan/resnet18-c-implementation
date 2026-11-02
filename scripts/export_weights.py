"""Export official ImageNet ResNet-18 weights to the C runtime format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.models import ResNet18_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1
TENSOR_COUNT = 42
sys.path.insert(0, str(PROJECT_ROOT))

from pytorch.model import ResNet18


def fold_batch_norm(
    conv: torch.nn.Conv2d, batch_norm: torch.nn.BatchNorm2d
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = batch_norm.weight / torch.sqrt(batch_norm.running_var + batch_norm.eps)
    weight = conv.weight * scale.reshape(-1, 1, 1, 1)
    bias = batch_norm.bias - batch_norm.running_mean * scale
    return weight, bias


def write_tensor(output, tensor: torch.Tensor, name: str, manifest: list[dict]) -> None:
    tensor = tensor.detach().cpu().contiguous().float()
    data = tensor.numpy().tobytes()
    manifest.append(
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": "float32",
            "offset": output.tell(),
            "bytes": len(data),
        }
    )
    output.write(data)


def export_weights(model: ResNet18, output_path: Path, manifest_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    payload_bytes = 0
    with torch.no_grad(), output_path.open("wb") as output:
        output.write(b"RN18")
        output.write((FORMAT_VERSION).to_bytes(4, "little"))
        output.write((TENSOR_COUNT).to_bytes(4, "little"))
        output.write((0).to_bytes(4, "little"))
        weight, bias = fold_batch_norm(model.conv1, model.bn1)
        write_tensor(output, weight, "conv1.weight_folded", manifest)
        write_tensor(output, bias, "conv1.bias_folded", manifest)

        for layer_index, layer in enumerate(
            (model.layer1, model.layer2, model.layer3, model.layer4), 1
        ):
            for block_index, block in enumerate(layer):
                prefix = f"layer{layer_index}.{block_index}"
                for conv, batch_norm in (
                    (block.conv1, block.bn1),
                    (block.conv2, block.bn2),
                ):
                    weight, bias = fold_batch_norm(conv, batch_norm)
                    conv_name = "conv1" if conv is block.conv1 else "conv2"
                    write_tensor(
                        output, weight, f"{prefix}.{conv_name}.weight_folded", manifest
                    )
                    write_tensor(
                        output, bias, f"{prefix}.{conv_name}.bias_folded", manifest
                    )
                if block.downsample is not None:
                    weight, bias = fold_batch_norm(
                        block.downsample[0], block.downsample[1]
                    )
                    write_tensor(
                        output, weight, f"{prefix}.downsample.weight_folded", manifest
                    )
                    write_tensor(
                        output, bias, f"{prefix}.downsample.bias_folded", manifest
                    )

        write_tensor(output, model.fc.weight, "fc.weight", manifest)
        write_tensor(output, model.fc.bias, "fc.bias", manifest)
        payload_bytes = output.tell() - 16
        output.seek(12)
        output.write(payload_bytes.to_bytes(4, "little"))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "format": "RN18",
                "version": FORMAT_VERSION,
                "endianness": "little",
                "header_bytes": 16,
                "tensor_count": len(manifest),
                "payload_bytes": payload_bytes,
                "tensors": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_classes(classes_path: Path, weights: ResNet18_Weights) -> None:
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    classes_path.write_text("\n".join(weights.meta["categories"]) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "weights" / "resnet18.bin",
        help="destination for the C-compatible binary weights",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "imagenet_classes.txt",
        help="destination for ImageNet class labels",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "weights" / "resnet18.json",
        help="destination for the tensor manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = ResNet18()
    state_dict = weights.get_state_dict(progress=True)
    model.load_state_dict(state_dict)
    model.eval()
    export_weights(model, args.output, args.manifest)
    write_classes(args.classes, weights)
    print(f"wrote {args.output}")
    print(f"wrote {args.classes}")
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
