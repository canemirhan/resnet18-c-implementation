# ResNet-18 Diagram-to-Code Mapping

This document maps the blocks in the ResNet-18 diagram to the C
implementation in this repository.

The diagram uses the names `Conv_1` through `Conv_5`. In the source code,
`Conv_1` is stored as `m->conv1`, while `Conv_2` through `Conv_5` correspond
to the four entries of `m->blocks[stage][block]`.

The implementation is in [src/resnet18.c](../src/resnet18.c), and the basic
operators are in [src/layers.c](../src/layers.c).

## Complete execution path

```text
Input image
  -> load_image_chw() or load_input_tensor()
  -> 3 x 224 x 224 normalized CHW tensor
  -> Conv_1: 7 x 7 convolution, 64 channels, stride 2
  -> 3 x 3 max pooling, stride 2
  -> Conv_2: two 64-channel residual blocks
  -> Conv_3: two 128-channel residual blocks
  -> Conv_4: two 256-channel residual blocks
  -> Conv_5: two 512-channel residual blocks
  -> global average pooling
  -> fully connected layer with 1000 outputs
  -> softmax and top-10 display
```

The model forward function ends at the 1000 raw logits. Softmax is performed
later by `main()` so that raw logits can optionally be saved for comparison
with PyTorch.

## Diagram labels and source-code locations

| Diagram block | Mathematical operation | C representation |
|---|---|---|
| Image | Preprocessed RGB input | `load_image_chw()` in `src/image_io.c` |
| `Conv_1` | `7x7`, 3 input channels, 64 output channels, stride 2, padding 3 | `m->conv1`; loaded by `read_conv(f, &m->conv1, 3, 64, 7, 2, 3)` |
| Pool | `3x3` max pool, stride 2, padding 1 | `maxpool3x3s2()` |
| `Conv_2` | Two residual blocks with 64 channels | `m->blocks[0][0]` and `m->blocks[0][1]` |
| `Conv_3` | Two residual blocks with 128 channels; first block downsamples | `m->blocks[1][0]` and `m->blocks[1][1]` |
| `Conv_4` | Two residual blocks with 256 channels; first block downsamples | `m->blocks[2][0]` and `m->blocks[2][1]` |
| `Conv_5` | Two residual blocks with 512 channels; first block downsamples | `m->blocks[3][0]` and `m->blocks[3][1]` |
| `7x7 Pooling, `/7` | Average every spatial position in each channel | `global_avgpool()` |
| `FC 1000` | 512 features to 1000 class logits | `fc_forward()` |
| `Softmax` | Convert logits into probabilities | Softmax loop in `src/main.c` |

The diagram labels `Conv_2`–`Conv_5` describe stages, not individual
convolution calls. Each stage contains two `BasicBlock` objects, and each
ordinary block contains two `3x3` convolutions.

## Input and stem: `Conv_1`

The C entry point is [main](../src/main.c), which chooses one of two input
paths:

```c
float *x = strcmp(argv[2], "--raw-input") == 0
    ? load_input_tensor(argv[3])
    : load_image_chw(argv[2]);
```

Both paths must produce the same logical tensor:

```text
3 x 224 x 224, FP32, CHW, ImageNet-normalized
```

The first model operation is in `resnet18_forward()`:

```c
conv_forward(&m->conv1, x, 224, 224, buf[0], &H, &W, NULL, 1);
```

This corresponds to the diagram's green block:

```text
7x7 conv, 64 channels, stride 2
224 x 224 x 3 -> 112 x 112 x 64
```

The output-size formula used by `conv_forward()` is:

```text
output = floor((input + 2 * padding - kernel) / stride) + 1
```

For the first convolution:

```text
floor((224 + 2 * 3 - 7) / 2) + 1
= floor(223 / 2) + 1
= 111 + 1
= 112
```

The final argument `1` enables ReLU after the convolution.

## Stem pooling

Immediately after `Conv_1`, the code calls:

```c
maxpool3x3s2(buf[0], C, H, W, buf[1], &H, &W);
```

This is the diagram's `Pool, /2` block. It uses:

```text
kernel = 3 x 3
stride = 2
padding = 1
```

The dimensions change from:

```text
112 x 112 x 64 -> 56 x 56 x 64
```

Max pooling does not change the channel count. For every channel, it selects
the largest value in each `3x3` window.

## What a BasicBlock means

The structure declaration is in [include/resnet18.h](../include/resnet18.h):

```c
typedef struct { Conv c1, c2, down; int has_down; } BasicBlock;
```

The fields correspond to the diagram as follows:

| Field | Meaning |
|---|---|
| `c1` | First `3x3` convolution in the block |
| `c2` | Second `3x3` convolution in the block |
| `down` | Optional `1x1` shortcut convolution |
| `has_down` | Whether the shortcut convolution is present |

The forward logic is in `block_forward()`:

```c
conv_forward(&b->c1, in, *H, *W, t1, &h1, &w1, NULL, 1);

const float *res = in;
if (b->has_down) {
    conv_forward(&b->down, in, *H, *W, sc, &hd, &wd, NULL, 0);
    res = sc;
}

conv_forward(&b->c2, t1, h1, w1, out, H, W, res, 1);
```

The operations are:

```text
first convolution + bias + ReLU
        ↓
second convolution + bias
        ↓
shortcut projection, if required
        ↓
main path + shortcut
        ↓
ReLU
```

The residual addition is implemented by passing `res` to the second
`conv_forward()` call. Inside `conv_forward()`:

```c
if (residual) acc += residual[out_index];
if (relu && acc < 0.f) acc = 0.f;
```

Therefore, the second convolution performs both the residual addition and
the final ReLU.

## `Conv_2`: first residual stage

In the diagram, `Conv_2` contains four visible `3x3 conv 64` boxes because
there are two BasicBlocks and each block has two convolutions.

The loader creates these blocks with `l = 0`:

```text
stage l = 0
out_c = 64
block 0: 64 -> 64, stride 1
block 1: 64 -> 64, stride 1
```

The spatial dimensions remain unchanged:

```text
56 x 56 x 64
    -> 56 x 56 x 64
    -> 56 x 56 x 64
```

Neither block needs a projection shortcut because the input and output shapes
are identical. The residual branch is simply the original `in` pointer.

## `Conv_3`: 128-channel residual stage

The diagram's first blue block is the first block of stage `l = 1`.

```text
stage l = 1
out_c = 128
```

The first block changes both spatial size and channel count:

```text
main path:
3x3, 64 -> 128, stride 2
3x3, 128 -> 128, stride 1

shortcut:
1x1, 64 -> 128, stride 2
```

The shortcut is needed because the residual addition requires matching shapes:

```text
56 x 56 x 64 -> 28 x 28 x 128
```

The second block keeps the shape:

```text
3x3, 128 -> 128, stride 1
3x3, 128 -> 128, stride 1
```

The complete stage is:

```text
56 x 56 x 64
    -> 28 x 28 x 128  (first block, downsampling)
    -> 28 x 28 x 128  (second block)
```

## `Conv_4`: 256-channel residual stage

Stage `l = 2` is the diagram's pink section:

```text
stage l = 2
out_c = 256
```

The first block uses:

```text
main path:
3x3, 128 -> 256, stride 2
3x3, 256 -> 256, stride 1

shortcut:
1x1, 128 -> 256, stride 2
```

The dimensions are:

```text
28 x 28 x 128
    -> 14 x 14 x 256  (first block)
    -> 14 x 14 x 256  (second block)
```

## `Conv_5`: 512-channel residual stage

Stage `l = 3` is the diagram's gray section:

```text
stage l = 3
out_c = 512
```

The first block uses:

```text
main path:
3x3, 256 -> 512, stride 2
3x3, 512 -> 512, stride 1

shortcut:
1x1, 256 -> 512, stride 2
```

The dimensions are:

```text
14 x 14 x 256
    -> 7 x 7 x 512  (first block)
    -> 7 x 7 x 512  (second block)
```

## Stage construction in the loader

The four stages are created by this nested loop in `resnet18_load()`:

```c
for (int l = 0; ok && l < 4; l++) {
    int out_c = 64 << l;
    for (int b = 0; ok && b < 2; b++) {
        int down = (b == 0 && l > 0);
        int in_c = down ? out_c / 2 : out_c;
        int stride = down ? 2 : 1;
        ...
    }
}
```

The variable meanings are:

| Variable | Meaning |
|---|---|
| `l` | Stage index: `0` to `3` |
| `b` | Block index inside a stage: `0` or `1` |
| `out_c` | Stage output channels: `64`, `128`, `256`, `512` |
| `down` | True only for the first block of stages `1`, `2`, and `3` |
| `in_c` | Input channels of the block |
| `stride` | `2` for stage transitions, otherwise `1` |

The expression:

```c
int out_c = 64 << l;
```

means:

```text
l = 0: 64  << 0 = 64
l = 1: 64  << 1 = 128
l = 2: 64  << 2 = 256
l = 3: 64  << 3 = 512
```

## Forward-pass dimensions

| Location | Height | Width | Channels | C buffer |
|---|---:|---:|---:|---|
| Input | 224 | 224 | 3 | `x` |
| After `Conv_1` | 112 | 112 | 64 | `buf[0]` |
| After max pool | 56 | 56 | 64 | `buf[1]` |
| After `Conv_2` | 56 | 56 | 64 | rotating buffer |
| After first block of `Conv_3` | 28 | 28 | 128 | rotating buffer |
| After `Conv_3` | 28 | 28 | 128 | rotating buffer |
| After first block of `Conv_4` | 14 | 14 | 256 | rotating buffer |
| After `Conv_4` | 14 | 14 | 256 | rotating buffer |
| After first block of `Conv_5` | 7 | 7 | 512 | rotating buffer |
| After `Conv_5` | 7 | 7 | 512 | rotating buffer |
| After global average pooling | 1 | 1 | 512 | `feat[512]` |
| After FC | 1 | 1 | 1000 | `logits[1000]` |

The `buf[0]` through `buf[3]` arrays are reusable activation buffers. The
integer `cur` identifies which buffer currently contains the input activation;
it is a buffer number, not a tensor channel or spatial index.

## Global average pooling

The diagram shows `7x7 Pooling, /7` because the final feature map is
`7x7x512`. The C implementation is more general:

```c
global_avgpool(buf[cur], C, H, W, feat);
```

For each channel `c`, it sums all `H * W` values and divides by `H * W`:

```text
feat[c] =
    (activation[c,0,0] + ... + activation[c,H-1,W-1]) / (H * W)
```

For the final ResNet-18 map:

```text
H = 7
W = 7
H * W = 49
```

Thus, each of the 512 channels becomes one scalar:

```text
7 x 7 x 512 -> 1 x 1 x 512
```

## Fully connected layer

The FC layer is called as:

```c
fc_forward(m->fc_w, m->fc_b, feat, 512, 1000, logits);
```

For class `o`, the calculation is:

```text
logits[o] = fc_b[o] + Σ(fc_w[o, i] * feat[i])
```

where:

```text
o = 0 ... 999
i = 0 ... 511
```

The flat weight index is:

```c
weight_index = o * in_n + i;
```

For example, class `2` uses weights:

```text
fc_w[2 * 512 + 0] ... fc_w[2 * 512 + 511]
```

The result is 1000 raw logits, one for each ImageNet class.

## Softmax and output

`resnet18_forward()` returns logits only. In [src/main.c](../src/main.c),
softmax is calculated using the numerically stable form:

```c
mx = maximum(logits)
logits[i] = expf(logits[i] - mx)
probability[i] = logits[i] / sum
```

Subtracting `mx` does not change the final probabilities, because the same
constant is subtracted from every logit. It prevents `expf()` from
overflowing when logits are large.

The program then repeatedly selects the largest probability to print the
top-10 classes. The selected value is temporarily set to `-1`, so the next
iteration finds the next-largest class.

## Important distinction: model output vs. display output

The pipeline has three different output representations:

1. **Feature map**: the final `7x7x512` activation before pooling.
2. **Logits**: the 1000 raw values returned by `resnet18_forward()`.
3. **Probabilities**: the softmax values calculated in `main()`.

The raw logits are the correct values for a numerical C-vs-PyTorch model
comparison. Probabilities are useful for user-facing classification output,
but they can hide or compress small numerical differences.
