#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "resnet18.h"

#define MAX_ACT (64 * 112 * 112)
/* Dört ortak tamponun her biri, en büyük ara aktivasyon olan 64x112x112
 * eleman için ayrılır; daha küçük aşamalar aynı tamponları yeniden kullanır. */

static int read_conv(FILE *f, Conv *c, int in_c, int out_c, int k, int stride, int pad)
{
    /*
     * f: binary dosya akışı.
     * c: doldurulacak konvolüsyon katmanı.
     * in_c/out_c: giriş ve çıkış kanal sayıları.
     * k: kernel kenar uzunluğu; stride ve pad konvolüsyon ayarlarıdır.
     */
    c->in_c = in_c; c->out_c = out_c; c->k = k; c->stride = stride; c->pad = pad;
    /* OIHW düzenindeki toplam ağırlık sayısıdır. */
    size_t nw = (size_t)out_c * in_c * k * k;
    c->w = malloc(nw * sizeof(float));
    c->b = malloc(out_c * sizeof(float));
    if (!c->w || !c->b) {
        free(c->w);
        free(c->b);
        c->w = NULL;
        c->b = NULL;
        return -1;
    }
    if (fread(c->w, sizeof(float), nw, f) != nw ||
        fread(c->b, sizeof(float), out_c, f) != (size_t)out_c) {
        free(c->w);
        free(c->b);
        c->w = NULL;
        c->b = NULL;
        return -1;
    }
    return 0;
}

ResNet18 *resnet18_load(const char *path)
{
    /* path, export_weights.py tarafından oluşturulan RN18 dosyasının yoludur. */
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    /* Header: 4 byte magic + version + tensor sayısı + payload boyutu. */
    char magic[4];
    uint32_t version, tensor_count, payload_bytes;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "RN18", 4) ||
        fread(&version, sizeof version, 1, f) != 1 ||
        fread(&tensor_count, sizeof tensor_count, 1, f) != 1 ||
        fread(&payload_bytes, sizeof payload_bytes, 1, f) != 1 ||
        version != 1 || tensor_count != 42) {
        fprintf(stderr, "bad weight file\n"); fclose(f); return NULL;
    }
    if (payload_bytes != 46738848u) {
        fprintf(stderr, "unexpected weight payload size\n");
        fclose(f);
        return NULL;
    }
    ResNet18 *m = calloc(1, sizeof *m);
    if (!m) {
        fprintf(stderr, "out of memory while loading model\n");
        fclose(f);
        return NULL;
    }
    /* İlk katman: RGB 3 kanaldan 64 kanala, 7x7 kernel, stride 2. */
    int ok = read_conv(f, &m->conv1, 3, 64, 7, 2, 3) == 0;

    for (int l = 0; ok && l < 4; l++) {
        /* l=0,1,2,3 için kanal sayısı 64,128,256,512 olur. */
        int out_c = 64 << l;
        for (int b = 0; ok && b < 2; b++) {
            /* Her stage iki BasicBlock içerir; stage başındaki bloklar
             * l>0 durumunda downsample shortcut kullanır. */
            int down = (b == 0 && l > 0);
            int in_c = down ? out_c / 2 : out_c;
            int stride = down ? 2 : 1;
            BasicBlock *blk = &m->blocks[l][b];
            ok = read_conv(f, &blk->c1, in_c, out_c, 3, stride, 1) == 0
              && read_conv(f, &blk->c2, out_c, out_c, 3, 1, 1) == 0;
            blk->has_down = down;
            if (ok && down) ok = read_conv(f, &blk->down, in_c, out_c, 1, 2, 0) == 0;
        }
    }
    if (ok) {
        /* FC ağırlığı 1000 sınıf x 512 özellik matrisidir. */
        m->fc_w = malloc(1000 * 512 * sizeof(float));
        m->fc_b = malloc(1000 * sizeof(float));
        ok = fread(m->fc_w, sizeof(float), 1000 * 512, f) == 1000 * 512
          && fread(m->fc_b, sizeof(float), 1000, f) == 1000;
    }
    /* Dosyanın tamamı tüketilmelidir; fazladan byte export/loader uyuşmazlığını gösterir. */
    if (ok && fgetc(f) != EOF) { fprintf(stderr, "trailing data in weights\n"); ok = 0; }
    fclose(f);
    if (!ok) { fprintf(stderr, "failed to load weights\n"); resnet18_free(m); return NULL; }
    return m;
}

static void block_forward(const BasicBlock *b, const float *in, float *t1, float *sc,
                          float *out, int *C, int *H, int *W)
{
    /*
     * in: blok girişi; t1: ilk konvolüsyon geçici çıktısı.
     * sc: downsample shortcut çıktısı; out: blok sonucu.
     * C/H/W pointer'ları blok sonrasındaki boyutlarla güncellenir.
     */
    int h1, w1, hd, wd;
    conv_forward(&b->c1, in, *H, *W, t1, &h1, &w1, NULL, 1);
    /* Residual, shortcut olmayan blokta doğrudan giriş aktivasyonudur. */
    const float *res = in;
    if (b->has_down) {
        conv_forward(&b->down, in, *H, *W, sc, &hd, &wd, NULL, 0);
        res = sc;
    }
    conv_forward(&b->c2, t1, h1, w1, out, H, W, res, 1);
    *C = b->c2.out_c;
}

void resnet18_forward(const ResNet18 *m, const float *x, float *logits)
{
    /* buf[0..3], büyük ara aktivasyonlar için yeniden kullanılan tamponlardır. */
    float *buf[4];
    for (int i = 0; i < 4; i++) {
        buf[i] = malloc(MAX_ACT * sizeof(float));
        if (!buf[i]) {
            for (int j = 0; j < i; j++) free(buf[j]);
            fprintf(stderr, "out of memory during ResNet-18 forward\n");
            return;
        }
    }

    int C = 64, H, W;
    /* x: 3x224x224 -> conv1 sonrası buf[0]: 64x112x112. */
    conv_forward(&m->conv1, x, 224, 224, buf[0], &H, &W, NULL, 1);
    /* maxpool sonrası buf[1]: 64x56x56. */
    maxpool3x3s2(buf[0], C, H, W, buf[1], &H, &W);

    /* cur, mevcut giriş aktivasyonunun buf dizisindeki tampon indeksidir. */
    int cur = 1;
    for (int l = 0; l < 4; l++)
        for (int b = 0; b < 2; b++) {
            int o[3], k = 0;
            /*
             * Dört tamponun biri giriş (cur), kalan üçü geçici çıktı
             * olarak kullanılır. o[k] değerleri tampon dizisinin indeksidir;
             * tensor içindeki kanal/satır/sütun indeksi değildir.
             */
            for (int i = 0; i < 4; i++) if (i != cur) o[k++] = i;
            block_forward(&m->blocks[l][b], buf[cur], buf[o[0]], buf[o[1]], buf[o[2]],
                          &C, &H, &W);
            cur = o[2];
        }

    /* Son 512 kanallı aktivasyon global average pooling ile 512 özelliğe iner. */
    float feat[512];
    global_avgpool(buf[cur], C, H, W, feat);
    fc_forward(m->fc_w, m->fc_b, feat, 512, 1000, logits);
    for (int i = 0; i < 4; i++) free(buf[i]);
}

void resnet18_free(ResNet18 *m)
{
    /* Model ve ona ait bütün heap bellekleri burada serbest bırakılır. */
    if (!m) return;
    free(m->conv1.w); free(m->conv1.b);
    for (int l = 0; l < 4; l++)
        for (int b = 0; b < 2; b++) {
            BasicBlock *k = &m->blocks[l][b];
            free(k->c1.w); free(k->c1.b); free(k->c2.w); free(k->c2.b);
            if (k->has_down) { free(k->down.w); free(k->down.b); }
        }
    free(m->fc_w); free(m->fc_b); free(m);
}