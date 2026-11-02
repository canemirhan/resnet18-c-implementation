#ifndef LAYERS_H
#define LAYERS_H

typedef struct { int in_c, out_c, k, stride, pad; float *w, *b; } Conv;

/*
 * out = conv(in) + bias [+ residual] [ReLU].
 * Tensorlar batch=1 ve CHW düzenindedir:
 * indeks = (kanal * yükseklik + satır) * genişlik + sütun.
 */
void conv_forward(const Conv *c, const float *in, int H, int W,
                  float *out, int *Ho, int *Wo,
                  const float *residual, int relu);
void maxpool3x3s2(const float *in, int C, int H, int W, float *out, int *Ho, int *Wo);
void global_avgpool(const float *in, int C, int H, int W, float *out);
void fc_forward(const float *w, const float *b, const float *in,
                int in_n, int out_n, float *out);
#endif