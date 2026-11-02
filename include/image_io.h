#ifndef IMAGE_IO_H
#define IMAGE_IO_H
/*
 * Görüntüyü RGB'ye çevirir, 256 kısa kenara resize eder, 224x224 merkez
 * kırpar ve ImageNet mean/std ile normalize edilmiş CHW tensörü döndürür.
 */
float *load_image_chw(const char *path);
#endif