#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "resnet18.h"
#include "image_io.h"
#include "tensor_io.h"

int main(int argc, char **argv)
{
    /*
     * argc: komut satırındaki toplam parça sayısı.
     * argv[0]: program adı.
     * argv[1]: RN18 ağırlık dosyası.
     * argv[2]: görüntü yolu veya --raw-input seçeneği.
     * argv[3]: görüntü yolu ya da ham tensör yolu.
     * argv[4]: sınıf isimleri dosyası.
     * argv[5]: isteğe bağlı ham logits çıktı dosyası.
     */
    if (argc < 4 ||
        (strcmp(argv[2], "--raw-input") == 0 && argc < 5)) {
        fprintf(stderr, "usage: %s weights.bin image.jpg classes.txt [logits.bin]\n"
                        "   or: %s weights.bin --raw-input input.bin classes.txt [logits.bin]\n"
                        "hint: use scripts/preprocess_image.py for exact torchvision preprocessing\n",
                argv[0], argv[0]);
        return 1;
    }
    /* m, ağırlıkları ve model katmanlarını tutan heap belleğidir. */
    ResNet18 *m = resnet18_load(argv[1]);
    /* x, modelin beklediği normalize edilmiş 3x224x224 CHW girişidir. */
    float *x = strcmp(argv[2], "--raw-input") == 0
        ? load_input_tensor(argv[3])
        : load_image_chw(argv[2]);
    if (!m || !x) return 1;

    /* labels[sinif_indeksi], o ImageNet sınıfının adıdır. */
    static char labels[1000][128];
    const char *classes_path = strcmp(argv[2], "--raw-input") == 0 ? argv[4] : argv[3];
    FILE *lf = fopen(classes_path, "r");
    if (!lf) { perror(classes_path); free(x); resnet18_free(m); return 1; }
    /* labels[i], ImageNet sınıf indeks i'nin metin karşılığıdır. */
    for (int i = 0; i < 1000 && fgets(labels[i], 128, lf); i++)
        labels[i][strcspn(labels[i], "\r\n")] = 0;
    fclose(lf);

    /* logits[i], i indeksli sınıfın softmax öncesi ham skorudur. */
    float logits[1000];
    resnet18_forward(m, x, logits);
    const char *logits_path =
        strcmp(argv[2], "--raw-input") == 0
            ? (argc >= 6 ? argv[5] : NULL)
            : (argc >= 5 ? argv[4] : NULL);
    if (logits_path && write_logits_tensor(logits_path, logits) != 0) {
        free(x);
        resnet18_free(m);
        return 1;
    }

    /* mx: taşmayı önlemek için çıkarılan en büyük ham skor.
     * sum: softmax paydası; olasılıkların toplamı bununla normalize edilir. */
    float mx = logits[0], sum = 0.f;
    /* Softmax öncesi maksimumu çıkarmak taşma riskini azaltır. */
    for (int i = 1; i < 1000; i++) if (logits[i] > mx) mx = logits[i];
    for (int i = 0; i < 1000; i++) { logits[i] = expf(logits[i] - mx); sum += logits[i]; }
    for (int i = 0; i < 1000; i++) logits[i] /= sum;

    for (int r = 0; r < 10; r++) {           /* r: sonuç sırası, 0'dan 9'a. */
        int best = 0;
        for (int i = 1; i < 1000; i++) if (logits[i] > logits[best]) best = i;
        /* best, logits dizisinin indeksidir; labels[best] sınıf adını verir. */
        printf("%d. %-30s %6.4f%%\n", r + 1, labels[best], logits[best] * 100.f);
        logits[best] = -1.f;
    }
    free(x); resnet18_free(m);
    return 0;
}