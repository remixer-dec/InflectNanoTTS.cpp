#include "ggml.h"
#include "ggml-quants.h"

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fputc('\n', stderr);
    abort();
}

static int64_t local_block_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return 256;
        case GGML_TYPE_Q1_0:
            return 128;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
            return 32;
        default:
            return 1;
    }
}

size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K: return 84;
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_Q1_0: return 18;
        case GGML_TYPE_Q4_0: return 18;
        case GGML_TYPE_Q5_0: return 22;
        case GGML_TYPE_Q8_0: return 34;
        default: return 0;
    }
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    return ggml_type_size(type) * (size_t)ne / (size_t)local_block_size(type);
}

const char * ggml_type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K: return "q2_K";
        case GGML_TYPE_Q3_K: return "q3_K";
        case GGML_TYPE_Q4_K: return "q4_K";
        case GGML_TYPE_Q5_K: return "q5_K";
        case GGML_TYPE_Q6_K: return "q6_K";
        case GGML_TYPE_Q1_0: return "q1_0";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q5_0: return "q5_0";
        case GGML_TYPE_Q8_0: return "q8_0";
        default: return "unknown";
    }
}

static enum ggml_type parse_type(const char * name) {
    if (strcmp(name, "q1_0") == 0) return GGML_TYPE_Q1_0;
    if (strcmp(name, "q1_k") == 0) return GGML_TYPE_Q1_0;
    if (strcmp(name, "q2_k") == 0) return GGML_TYPE_Q2_K;
    if (strcmp(name, "q3_k") == 0) return GGML_TYPE_Q3_K;
    if (strcmp(name, "q4_k") == 0) return GGML_TYPE_Q4_K;
    if (strcmp(name, "q5_k") == 0) return GGML_TYPE_Q5_K;
    if (strcmp(name, "q6_k") == 0) return GGML_TYPE_Q6_K;
    if (strcmp(name, "q4_0") == 0) return GGML_TYPE_Q4_0;
    if (strcmp(name, "q5_0") == 0) return GGML_TYPE_Q5_0;
    if (strcmp(name, "q8_0") == 0) return GGML_TYPE_Q8_0;
    return GGML_TYPE_COUNT;
}

static size_t quantize_rows(enum ggml_type type, const float * input, void * output, int64_t rows, int64_t cols) {
    switch (type) {
        case GGML_TYPE_Q2_K: return quantize_q2_K(input, output, rows, cols, NULL);
        case GGML_TYPE_Q3_K: return quantize_q3_K(input, output, rows, cols, NULL);
        case GGML_TYPE_Q4_K: return quantize_q4_K(input, output, rows, cols, NULL);
        case GGML_TYPE_Q5_K: return quantize_q5_K(input, output, rows, cols, NULL);
        case GGML_TYPE_Q6_K: return quantize_q6_K(input, output, rows, cols, NULL);
        case GGML_TYPE_Q1_0: return quantize_q1_0(input, output, rows, cols, NULL);
        case GGML_TYPE_Q4_0: return quantize_q4_0(input, output, rows, cols, NULL);
        case GGML_TYPE_Q5_0: return quantize_q5_0(input, output, rows, cols, NULL);
        case GGML_TYPE_Q8_0: return quantize_q8_0(input, output, rows, cols, NULL);
        default: return 0;
    }
}

static int parse_i64(const char * s, int64_t * out) {
    char * end = NULL;
    errno = 0;
    long long v = strtoll(s, &end, 10);
    if (errno || end == s || *end != '\0' || v <= 0) return 0;
    *out = (int64_t)v;
    return 1;
}

static int read_exact(void * dst, size_t n) {
    uint8_t * p = (uint8_t *)dst;
    while (n > 0) {
        size_t got = fread(p, 1, n, stdin);
        if (got == 0) return 0;
        p += got;
        n -= got;
    }
    return 1;
}

static int write_exact(const void * src, size_t n) {
    const uint8_t * p = (const uint8_t *)src;
    while (n > 0) {
        size_t wrote = fwrite(p, 1, n, stdout);
        if (wrote == 0) return 0;
        p += wrote;
        n -= wrote;
    }
    return 1;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s TYPE ROWS COLS\n", argv[0]);
        return 2;
    }

    enum ggml_type type = parse_type(argv[1]);
    int64_t rows = 0;
    int64_t cols = 0;
    if (type == GGML_TYPE_COUNT || !parse_i64(argv[2], &rows) || !parse_i64(argv[3], &cols)) {
        fprintf(stderr, "invalid arguments\n");
        return 2;
    }
    if (cols % local_block_size(type) != 0) {
        fprintf(stderr, "cols=%lld is not aligned to block size %lld for %s\n",
                (long long)cols, (long long)local_block_size(type), argv[1]);
        return 2;
    }

    const size_t in_count = (size_t)rows * (size_t)cols;
    const size_t in_bytes = in_count * sizeof(float);
    const size_t row_bytes = ggml_row_size(type, cols);
    const size_t out_bytes = (size_t)rows * row_bytes;

    float * input = (float *)malloc(in_bytes);
    void * output = malloc(out_bytes);
    if (!input || !output) {
        fprintf(stderr, "allocation failed\n");
        free(input);
        free(output);
        return 1;
    }

    if (!read_exact(input, in_bytes)) {
        fprintf(stderr, "failed to read %zu input bytes\n", in_bytes);
        free(input);
        free(output);
        return 1;
    }

    size_t got = quantize_rows(type, input, output, rows, cols);
    if (got != out_bytes) {
        fprintf(stderr, "quantized size mismatch: got=%zu expected=%zu\n", got, out_bytes);
        free(input);
        free(output);
        return 1;
    }

    int ok = write_exact(output, out_bytes);
    free(input);
    free(output);
    return ok ? 0 : 1;
}
