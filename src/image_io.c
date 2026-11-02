#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "image_io.h"

#define COEFF_BITS 22
#define COEFF_SCALE (1 << COEFF_BITS)

static double bilinear_filter(double x)
{
    /* x, kaynak piksel ile hedef koordinat arasındaki normalize uzaklıktır. */
    if (x < 0.0) x = -x;
    return x < 1.0 ? 1.0 - x : 0.0;
}

static int round_coefficient(double value)
{
    return value < 0.0
        ? (int)(-0.5 + value * COEFF_SCALE)
        : (int)(0.5 + value * COEFF_SCALE);
}

static int prepare_coefficients(int input_size, int output_size,
                                int **bounds_out, int **coefficients_out,
                                int *kernel_size_out)
{
    /*
     * input_size : bir eksendeki kaynak uzunluğu (width veya height).
     * output_size: aynı eksendeki hedef uzunluk.
     * scale      : kaynak/hedef oranı.
     * support    : bir hedef pikselin kullanacağı kaynak komşuluk yarıçapı.
     * kernel_size: katsayı tamponundaki her hedef piksel için ayrılan alan.
     */
    double scale = (double)input_size / output_size;
    double filter_scale = scale < 1.0 ? 1.0 : scale;
    double support = filter_scale;
    int kernel_size = (int)ceil(support) * 2 + 1;
    int *bounds = malloc((size_t)output_size * 2 * sizeof(*bounds));
    int *coefficients = calloc((size_t)output_size * kernel_size,
                               sizeof(*coefficients));

    if (!bounds || !coefficients) {
        free(bounds);
        free(coefficients);
        return -1;
    }

    for (int output = 0; output < output_size; output++) {
        /*
         * Pillow ile aynı koordinat hesabı:
         * center, kaynak piksel koordinat sistemindeki pencere merkezidir.
         */
        double center = (output + 0.5) * scale;
        /* first/last, bu hedef piksel için kaynak aralığının sınırlarıdır. */
        int first = (int)(center - support + 0.5);
        int last = (int)(center + support + 0.5);
        if (first < 0) first = 0;
        if (last > input_size) last = input_size;
        bounds[output * 2] = first;
        bounds[output * 2 + 1] = last - first;

        double total = 0.0;
        for (int i = 0; i < last - first; i++) {
            double distance = (i + first - center + 0.5) / filter_scale;
            double weight = bilinear_filter(distance);
            total += weight;
        }
        if (total != 0.0) {
            for (int i = 0; i < last - first; i++) {
                double distance = (i + first - center + 0.5) / filter_scale;
                coefficients[output * kernel_size + i] =
                    round_coefficient(bilinear_filter(distance) / total);
            }
        }
    }

    *bounds_out = bounds;
    *coefficients_out = coefficients;
    *kernel_size_out = kernel_size;
    return 0;
}

static int clip_fixed(int64_t value)
{
    /* Sabit noktalı toplamı tekrar 0..255 aralığındaki piksele çevirir. */
    int64_t rounded = value >= 0
        ? (value + (1LL << (COEFF_BITS - 1))) >> COEFF_BITS
        : (value - (1LL << (COEFF_BITS - 1))) >> COEFF_BITS;
    if (rounded < 0) return 0;
    if (rounded > 255) return 255;
    return (int)rounded;
}

static int resize_pillow_bilinear(const unsigned char *input, int width, int height,
                                  unsigned char *output, int output_width,
                                  int output_height, int channels)
{
    /*
     * input/output: HWC düzeninde uint8 görüntüler.
     * x_bounds/y_bounds: her hedef koordinat için kaynak pencere sınırları.
     * x_coefficients/y_coefficients: bilinear katsayıları.
     * intermediate: yatay geçişten sonraki geçici görüntü.
     */
    int *x_bounds, *x_coefficients, x_kernel;
    int *y_bounds, *y_coefficients, y_kernel;
    size_t intermediate_size = (size_t)output_width * height * channels;
    unsigned char *intermediate = malloc(intermediate_size);

    if (!intermediate ||
        prepare_coefficients(width, output_width, &x_bounds, &x_coefficients,
                             &x_kernel) != 0) {
        free(intermediate);
        return -1;
    }
    if (prepare_coefficients(height, output_height, &y_bounds, &y_coefficients,
                             &y_kernel) != 0) {
        free(intermediate);
        free(x_bounds);
        free(x_coefficients);
        return -1;
    }

    /*
     * Yatay geçiş: her satırdaki genişlik output_width olur.
     * Her kanal için ayrı bir sabit noktalı toplam hesaplanır.
     */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < output_width; x++) {
            /* x, hedef görüntünün sütun indeksi; y, satır indeksidir. */
            int first = x_bounds[x * 2];
            int count = x_bounds[x * 2 + 1];
            for (int c = 0; c < channels; c++) {
                /* c=0,1,2 sırasıyla kırmızı, yeşil, mavi kanal indeksidir. */
                int64_t sum = 1LL << (COEFF_BITS - 1);
                for (int i = 0; i < count; i++)
                    sum += (int64_t)input[(y * width + first + i) * channels + c] *
                           x_coefficients[x * x_kernel + i];
                intermediate[(y * output_width + x) * channels + c] =
                    (unsigned char)clip_fixed(sum);
            }
        }
    }

    /*
     * Dikey geçiş: intermediate yüksekliği height, output yüksekliği
     * output_height olur. Bu ikinci geçişten sonra sonuç son uint8 görüntüdür.
     */
    for (int y = 0; y < output_height; y++) {
        int first = y_bounds[y * 2];
        int count = y_bounds[y * 2 + 1];
        for (int x = 0; x < output_width; x++) {
            for (int c = 0; c < channels; c++) {
                int64_t sum = 1LL << (COEFF_BITS - 1);
                for (int i = 0; i < count; i++)
                    sum += (int64_t)intermediate[((first + i) * output_width + x) *
                                                 channels + c] *
                           y_coefficients[y * y_kernel + i];
                output[(y * output_width + x) * channels + c] =
                    (unsigned char)clip_fixed(sum);
            }
        }
    }

    free(intermediate);
    free(x_bounds);
    free(x_coefficients);
    free(y_bounds);
    free(y_coefficients);
    return 0;
}

float *load_image_chw(const char *path)
{
    /*
     * width/height: kaynak görüntünün boyutları.
     * channels: dosyadaki kanal sayısı; stbi_load sonucu RGB olarak istenir.
     * resized_width/resized_height: kısa kenarı 256 yapılmış boyutlar.
     * crop_x/crop_y: 224x224 merkez kırpmanın başlangıç koordinatı.
     */
    int width, height, channels;
    unsigned char *image = stbi_load(path, &width, &height, &channels, 3);
    if (!image) {
        fprintf(stderr, "cannot load %s\n", path);
        return NULL;
    }

    int resized_width, resized_height;
    if (width < height) {
        resized_width = 256;
        resized_height = (int)((double)height * 256 / width + 0.5);
    } else {
        resized_height = 256;
        resized_width = (int)((double)width * 256 / height + 0.5);
    }

    unsigned char *resized = malloc((size_t)resized_width * resized_height * 3);
    if (!resized ||
        resize_pillow_bilinear(image, width, height, resized, resized_width,
                               resized_height, 3) != 0) {
        fprintf(stderr, "out of memory while resizing %s\n", path);
        free(resized);
        stbi_image_free(image);
        return NULL;
    }
    stbi_image_free(image);

    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float standard_deviation[3] = {0.229f, 0.224f, 0.225f};
    int crop_x = (resized_width - 224) / 2;
    int crop_y = (resized_height - 224) / 2;
    /* output, son normalize edilmiş CHW tensörüdür. */
    float *output = malloc(3 * 224 * 224 * sizeof(*output));
    if (!output) {
        fprintf(stderr, "out of memory while loading %s\n", path);
        free(resized);
        return NULL;
    }

    for (int c = 0; c < 3; c++)
        for (int y = 0; y < 224; y++)
            for (int x = 0; x < 224; x++) {
                /* Önce HWC görüntüden piksel alınır, sonra CHW output'a yazılır. */
                float pixel = resized[((crop_y + y) * resized_width +
                                       (crop_x + x)) * 3 + c] / 255.0f;
                output[(c * 224 + y) * 224 + x] =
                    (pixel - mean[c]) / standard_deviation[c];
            }
    free(resized);
    return output;
}
