#ifndef TENSOR_IO_H
#define TENSOR_IO_H

/* RNIN header'lı 3x224x224 normalize edilmiş FP32 tensörü yükler. */
float *load_input_tensor(const char *path);
/* 1000 logits değerini RNLG header'lı FP32 dosyaya yazar. */
int write_logits_tensor(const char *path, const float *logits);

#endif
