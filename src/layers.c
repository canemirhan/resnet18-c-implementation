#include <float.h>
#include "layers.h"

void conv_forward(const Conv *c, const float *in, int H, int W,
                  float *out, int *Ho, int *Wo,
                  const float *residual, int relu)
{
    /*
     * H  : giriş aktivasyonunun yüksekliği (satır sayısı).
     * W  : giriş aktivasyonunun genişliği (sütun sayısı).
     * c->k     : kernel boyutu; ResNet'te genellikle 3 veya 7.
     * c->pad   : girişin etrafına eklenen sıfır padding miktarı.
     * c->stride: kernel penceresinin her adımda kaç piksel ilerlediği.
     *
     * oh (output height) : çıkış yüksekliği, yani çıkışta kaç satır vardır?
     * ow (output width)  : çıkış genişliği, yani çıkışta kaç sütun vardır?
     *
     * Örnek: H=W=224, k=7, pad=3, stride=2:
     * oh = ow = (224 + 2*3 - 7) / 2 + 1 = 112.
     * C'deki int bölmesi floor işlemi gibi davranır.
     */
    int oh = (H + 2 * c->pad - c->k) / c->stride + 1;
    int ow = (W + 2 * c->pad - c->k) / c->stride + 1;
    /* Çağıran fonksiyon sonraki katmanın boyutunu öğrenebilsin. */
    *Ho = oh; *Wo = ow;

    #pragma omp parallel for
    for (int oc = 0; oc < c->out_c; oc++) { /* oc: çıkış kanalı indeksi */
        for (int oy = 0; oy < oh; oy++) {    /* oy: çıkış satırı indeksi */
            for (int ox = 0; ox < ow; ox++) { /* ox: çıkış sütunu indeksi */
                /*
                 * acc (accumulator), bu çıkış elemanının toplamıdır.
                 * Önce bias eklenir; sonra bütün giriş kanalları ve kernel
                 * penceresi üzerinden çarpım-toplam yapılır.
                 */
                float acc = c->b[oc];
                for (int ic = 0; ic < c->in_c; ic++) { /* ic: giriş kanalı */
                    for (int ky = 0; ky < c->k; ky++) { /* ky: kernel satırı */
                        /*
                         * Giriş koordinatı. Örneğin stride=2 ve pad=1 ise
                         * oy=0, ky=0 için iy=-1 olur; bu değer padding alanıdır
                         * ve aşağıdaki sınır kontrolüyle atlanır.
                         */
                        int iy = oy * c->stride - c->pad + ky;
                        if (iy < 0 || iy >= H) continue;
                        for (int kx = 0; kx < c->k; kx++) { /* kx: kernel sütunu */
                            int ix = ox * c->stride - c->pad + kx;
                            if (ix < 0 || ix >= W) continue;

                            /*
                             * CHW düzleştirme:
                             *   (kanal, satır, sütun) -> (kanal*H + satır)*W + sütun
                             * Örnek: C=3,H=4,W=4 için (c=1,y=2,x=3)
                             * indeksi (1*4+2)*4+3 = 27 olur.
                             *
                             * Ağırlık düzeni OIHW'dir:
                             *   (çıkış kanalı, giriş kanalı, ky, kx)
                             * -> (((oc*in_c)+ic)*k+ky)*k+kx.
                             */
                            /*
                             * in_index: girişteki (ic, iy, ix) elemanının
                             * düz dizi indeksi.
                             * weight_index: ağırlıktaki (oc, ic, ky, kx)
                             * elemanının düz dizi indeksi.
                             */
                            int in_index = (ic * H + iy) * W + ix;
                            int weight_index =
                                ((oc * c->in_c + ic) * c->k + ky) * c->k + kx;
                            acc += in[in_index] * c->w[weight_index];
                        }
                    }
                }
                /*
                 * out_index: çıkıştaki (oc, oy, ox) elemanının indeksi.
                 * residual aynı boyuttaysa aynı indeksle toplanabilir.
                 */
                int out_index = (oc * oh + oy) * ow + ox;
                if (residual) acc += residual[out_index];
                if (relu && acc < 0.f) acc = 0.f;
                out[out_index] = acc;
            }
        }
    }
}

void maxpool3x3s2(const float *in, int C, int H, int W, float *out, int *Ho, int *Wo)
{
    /*
     * C: kanal sayısı; her kanal bağımsız olarak havuzlanır.
     * H/W: giriş yüksekliği/genişliği.
     * Kernel=3, stride=2, padding=1 olduğundan:
     * oh = (H + 2*1 - 3) / 2 + 1
     * ow = (W + 2*1 - 3) / 2 + 1
     * Örnek: H=W=112 ise oh=ow=56 olur.
     */
    int oh = (H + 2 - 3) / 2 + 1, ow = (W + 2 - 3) / 2 + 1;
    *Ho = oh; *Wo = ow;
    for (int c = 0; c < C; c++) { /* c: kanal indeksi */
        for (int oy = 0; oy < oh; oy++) { /* oy: çıkış satırı */
            for (int ox = 0; ox < ow; ox++) { /* ox: çıkış sütunu */
                float m = -FLT_MAX;
                for (int ky = 0; ky < 3; ky++) { /* ky: 3x3 pencere satırı */
                    int iy = oy * 2 - 1 + ky;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < 3; kx++) { /* kx: pencere sütunu */
                        int ix = ox * 2 - 1 + kx;
                        if (ix < 0 || ix >= W) continue;
                        /* v, c kanalındaki (iy,ix) pikselinin CHW indeksidir. */
                        int in_index = (c * H + iy) * W + ix;
                        float v = in[in_index];
                        if (v > m) m = v;
                    }
                }
                int out_index = (c * oh + oy) * ow + ox;
                out[out_index] = m;
            }
        }
    }
}

void global_avgpool(const float *in, int C, int H, int W, float *out)
{
    /*
     * Global average pooling, her kanalın H*W elemanını tek bir sayıya
     * indirger. Sonuçta out dizisinin uzunluğu C olur.
     */
    for (int c = 0; c < C; c++) { /* c: kanal indeksi */
        float s = 0.f;
        for (int i = 0; i < H * W; i++) { /* i: kanal içindeki düz indeks */
            int in_index = c * H * W + i;
            s += in[in_index];
        }
        out[c] = s / (float)(H * W);
    }
}

void fc_forward(const float *w, const float *b, const float *in,
                int in_n, int out_n, float *out)
{
    for (int o = 0; o < out_n; o++) { /* o: çıkış sınıfı/nöronu indeksi */
        float s = b[o];
        /*
         * FC ağırlıkları [out_n][in_n] matrisidir.
         * w[o*in_n+i], o sınıfının i'nci özellik ağırlığıdır.
         * Örnek: in_n=4 ve o=2 ise sınıf 2'nin ağırlıkları w[8..11]'dir.
         */
        for (int i = 0; i < in_n; i++) { /* i: giriş özelliği indeksi */
            int weight_index = o * in_n + i;
            s += w[weight_index] * in[i];
        }
        out[o] = s;
    }
}