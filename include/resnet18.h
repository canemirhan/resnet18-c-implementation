#ifndef RESNET18_H
#define RESNET18_H
#include "layers.h"

typedef struct { Conv c1, c2, down; int has_down; } BasicBlock;
typedef struct {
    /* conv1, ImageNet girişindeki ilk 7x7 konvolüsyondur. */
    Conv conv1;
    /* blocks[stage][block]: dört stage, her stage iki BasicBlock. */
    BasicBlock blocks[4][2];
    /* FC katmanının 1000x512 ağırlıkları ve 1000 bias değeri. */
    float *fc_w, *fc_b;
} ResNet18;

/* Binary RN18 dosyasını açar; hata durumunda NULL döndürür. */
ResNet18 *resnet18_load(const char *path);
/* x: 3x224x224 giriş; logits: 1000 sınıf skoru. */
void resnet18_forward(const ResNet18 *m, const float *x, float *logits);
/* Modelin bütün ağırlık ve aktivasyon belleğini serbest bırakır. */
void resnet18_free(ResNet18 *m);
#endif