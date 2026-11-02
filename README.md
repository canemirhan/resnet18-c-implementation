# ResNet-18 C Inference

This project contains:

- A PyTorch reference implementation of ImageNet ResNet-18.
- A portable FP32 C inference implementation.
- An exporter for the official `ResNet18_Weights.IMAGENET1K_V1` checkpoint.
- Direct image inference and exact raw-tensor PyTorch-vs-C comparison tools.

Batch normalization is folded into convolution weights during export. The C
runtime therefore executes convolution, ReLU, max pooling, residual shortcuts,
global average pooling, and the final fully connected layer.

## Project layout

```text
include/                 C public headers
src/                     C implementation
tests/                   C forward test
pytorch/                 PyTorch reference model and inference
scripts/                 Export, preprocessing, and comparison tools
weights/                 Exported FP32 weights and manifest
data/sample_images/      Sample input images
third_party/             stb_image.h
Makefile                 Native build and test targets
```

For a diagram-to-code explanation of every ResNet-18 block, tensor shape, and
shortcut connection, see
[docs/ARCHITECTURE_MAPPING.md](docs/ARCHITECTURE_MAPPING.md).

## Requirements

### Python

Create or activate a virtual environment and install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Required packages are PyTorch, torchvision, and Pillow.

### C toolchain

The Makefile requires GNU Make and a C compiler such as GCC. On Windows,
install MinGW/MSYS2 and make sure both `gcc` and `make` are available in
PowerShell or in the MSYS2 terminal.

Check the tools with:

```powershell
gcc --version
make --version
```

## Build with the Makefile

Run these commands from the repository root.

### Build the C executable

```powershell
make
```

The default `all` target builds `resnet18`. The Makefile compiles:

```text
src/image_io.c
src/tensor_io.c
src/layers.c
src/resnet18.c
src/main.c
```

The compiler settings can be overridden without editing the Makefile:

```powershell
make CC=gcc CFLAGS="-O2 -std=c11 -Wall -Wextra -Wpedantic"
```

### Build only the executable target

```powershell
make resnet18
```

### Build the forward test

```powershell
make test_forward
```

### Run the C test

```powershell
make test
```

The test loads `weights/resnet18.bin`, executes a deterministic FP32 input
through the complete network, and checks that all 1,000 logits are finite.

### Remove build artifacts

```powershell
make clean
```

On native Windows, the Makefile commands use Unix-style executable names
(`resnet18` and `test_forward`). If your MinGW environment produces `.exe`
files, run the generated `.exe` file when invoking the program manually.

If GNU Make is not installed, the same compilation can be performed directly
with the GCC available in MinGW:

```powershell
gcc -Iinclude -Ithird_party -O2 -std=c11 -Wall -Wextra -Wpedantic `
  src\image_io.c src\tensor_io.c src\layers.c src\resnet18.c src\main.c `
  -lm -o resnet18.exe
```

This is only a fallback; using `make` is preferred because it also supports
incremental object-file builds and the test targets. To run `make test` on
Windows, use an MSYS2/MinGW terminal if your native PowerShell make does not
provide a compatible Unix shell.

## Export the official pretrained weights

From the repository root:

```powershell
python scripts\export_weights.py
```

This downloads the official torchvision checkpoint if it is not cached and
creates:

```text
weights/resnet18.bin       C-compatible FP32 binary
weights/resnet18.json      tensor manifest
scripts/imagenet_classes.txt
```

The binary contains a 16-byte little-endian header followed by 42 contiguous
FP32 tensors in the order expected by `resnet18_load`:

```text
magic:          RN18
version:        1
tensor_count:   42
payload_bytes:  46738848
```

## Direct image inference in C

After building the executable and exporting the weights:

```powershell
.\resnet18.exe `
  .\weights\resnet18.bin `
  .\data\sample_images\images.jfif `
  .\scripts\imagenet_classes.txt
```

Arguments:

1. `weights/resnet18.bin`: exported model parameters.
2. `data/sample_images/images.jfif`: input image.
3. `scripts/imagenet_classes.txt`: ImageNet class names.

The C image loader:

1. Loads the image with `stb_image`.
2. Converts it to RGB.
3. Resizes the shorter side to 256 pixels.
4. Uses a two-pass antialiased bilinear resize.
5. Center-crops to `224x224`.
6. Converts pixels to CHW FP32.
7. Applies ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The executable prints the top-10 ImageNet predictions and probabilities.

An optional fourth argument writes the raw C logits:

```powershell
.\resnet18.exe `
  .\weights\resnet18.bin `
  .\data\sample_images\images.jfif `
  .\scripts\imagenet_classes.txt `
  .\artifacts\test_outputs\direct_c_logits.bin
```

## PyTorch reference inference

Run the reference implementation with the official torchvision preprocessing:

```powershell
python -m pytorch.inference `
  .\data\sample_images\images.jfif `
  --output-dir .\artifacts\test_outputs
```

The command prints the PyTorch top-10 results and writes:

```text
artifacts/test_outputs/input.pt
artifacts/test_outputs/pytorch_logits.pt
artifacts/test_outputs/input.bin
artifacts/test_outputs/pytorch_logits.bin
artifacts/test_outputs/predictions.json
```

## Exact raw-tensor comparison

This is the recommended test for the C model implementation itself. It removes
image decoder and resize differences by giving both implementations exactly
the same preprocessed `3x224x224` FP32 tensor.

Generate the torchvision-preprocessed tensor:

```powershell
python scripts\preprocess_image.py `
  .\data\sample_images\images.jfif `
  .\artifacts\test_outputs\input.bin
```

Generate PyTorch reference logits:

```powershell
python -m pytorch.inference `
  .\data\sample_images\images.jfif `
  --output-dir .\artifacts\test_outputs
```

Run C with the exact same tensor:

```powershell
.\resnet18.exe `
  .\weights\resnet18.bin `
  --raw-input .\artifacts\test_outputs\input.bin `
  .\scripts\imagenet_classes.txt `
  .\artifacts\test_outputs\c_logits.bin
```

Compare logits and rankings:

```powershell
python scripts\compare_logits.py `
  .\artifacts\test_outputs\pytorch_logits.bin `
  .\artifacts\test_outputs\c_logits.bin
```

The comparison reports:

- Maximum absolute error.
- Mean absolute error.
- Maximum relative error.
- Top-1 agreement.
- Top-5 agreement.
- Top-10 agreement.

The expected FP32 result is a very small logit error, usually around `1e-5`,
with matching top-1, top-5, and top-10 rankings.

## Compare direct C preprocessing with PyTorch

This test runs the image through C preprocessing directly, runs the same image
through torchvision preprocessing in PyTorch, and compares both model outputs:

```powershell
python scripts\compare_direct_image.py `
  .\resnet18.exe `
  .\weights\resnet18.bin `
  .\data\sample_images\images.jfif `
  .\scripts\imagenet_classes.txt
```

It reports logits errors, PyTorch and C probabilities for the top-10 classes,
and top-1/top-5/top-10 agreement. Small differences can remain because Pillow
and the C implementation may differ in coefficient rounding or intermediate
integer behavior.

## C tensor indexing

The C runtime stores tensors in CHW layout:

```text
index(channel, row, column) = (channel * height + row) * width + column
```

Convolution weights use OIHW layout:

```text
index(out_channel, in_channel, kernel_row, kernel_column)
    = (((out_channel * input_channels) + in_channel) * kernel
       + kernel_row) * kernel + kernel_column
```

The C comments in `src/layers.c`, `src/image_io.c`, `src/resnet18.c`, and
`src/main.c` explain the variables and indexing calculations in detail.

## Current validation

The complete C forward path has been compiled and tested with the exported
official weights. Raw-tensor comparison has produced:

```text
max absolute error: approximately 8e-06
mean absolute error: approximately 1e-06
top-1 agreement: True
top-5 agreement: True
top-10 agreement: True
```

Exact values may vary slightly with compiler, optimization level, and CPU.
