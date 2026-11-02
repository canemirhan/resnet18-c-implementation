#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "tensor_io.h"

static int read_header(FILE *file, const char magic[4], uint32_t count)
{
    /* Header'daki count, takip eden float sayısını belirtir; byte sayısı değil. */
    /* actual, dosyadan okunan magic değeridir; magic beklenen değerdir. */
    char actual[4];
    uint32_t version, actual_count, reserved;
    return fread(actual, 1, 4, file) == 4 &&
           fread(&version, sizeof version, 1, file) == 1 &&
           fread(&actual_count, sizeof actual_count, 1, file) == 1 &&
           fread(&reserved, sizeof reserved, 1, file) == 1 &&
           actual[0] == magic[0] && actual[1] == magic[1] &&
           actual[2] == magic[2] && actual[3] == magic[3] &&
           version == 1 && actual_count == count && reserved == 0;
}

static int write_header(FILE *file, const char magic[4], uint32_t count)
{
    /* magic dosya tipidir; count header'dan sonra gelecek float sayısıdır. */
    /* Header alanları little-endian uint32 olarak yazılır. */
    uint32_t zero = 0, version = 1;
    return fwrite(magic, 1, 4, file) == 4 &&
           fwrite(&version, sizeof version, 1, file) == 1 &&
           fwrite(&count, sizeof count, 1, file) == 1 &&
           fwrite(&zero, sizeof zero, 1, file) == 1;
}

float *load_input_tensor(const char *path)
{
    /* 3 kanal * 224 satır * 224 sütun = 150528 float. */
    const uint32_t count = 3u * 224u * 224u;
    FILE *file = fopen(path, "rb");
    float *tensor;
    if (!file) {
        perror(path);
        return NULL;
    }
    if (!read_header(file, "RNIN", count)) {
        fprintf(stderr, "invalid input tensor header: %s\n", path);
        fclose(file);
        return NULL;
    }
    /* Header'dan sonra count adet normalize edilmiş FP32 değer okunur. */
    tensor = malloc((size_t)count * sizeof(*tensor));
    if (!tensor || fread(tensor, sizeof(*tensor), count, file) != count ||
        fgetc(file) != EOF) {
        fprintf(stderr, "invalid input tensor payload: %s\n", path);
        free(tensor);
        fclose(file);
        return NULL;
    }
    fclose(file);
    return tensor;
}

int write_logits_tensor(const char *path, const float *logits)
{
    /* 1000 indeksin her biri bir ImageNet sınıfının ham skorudur. */
    /* logits, softmax uygulanmadan önceki 1000 FP32 sınıf skorudur. */
    FILE *file = fopen(path, "wb");
    int ok = file && write_header(file, "RNLG", 1000) &&
             fwrite(logits, sizeof(*logits), 1000, file) == 1000;
    if (file) fclose(file);
    if (!ok) {
        fprintf(stderr, "cannot write logits tensor: %s\n", path);
        return -1;
    }
    return 0;
}
