#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "resnet18.h"

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s weights.bin\n", argv[0]);
        return 2;
    }
    ResNet18 *model = resnet18_load(argv[1]);
    if (!model) return 1;
    float *input = malloc(3 * 224 * 224 * sizeof(*input));
    float logits[1000];
    if (!input) {
        resnet18_free(model);
        return 1;
    }
    for (int i = 0; i < 3 * 224 * 224; i++)
        input[i] = (float)((i % 251) - 125) / 125.f;
    resnet18_forward(model, input, logits);
    int finite = 1;
    for (int i = 0; i < 1000; i++) finite = finite && isfinite(logits[i]);
    free(input);
    resnet18_free(model);
    if (!finite) {
        fprintf(stderr, "forward produced non-finite logits\n");
        return 1;
    }
    puts("ResNet-18 forward: ok");
    return 0;
}
