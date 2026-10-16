/* Reference dequantizers for the GGUF (ggml) block formats and MLX affine
 * quantization - see h3_quant.h. The block layouts and the expansion
 * formulas follow ggml's ggml-common.h / ggml-quants.c
 * (dequantize_row_q*), which the GGUF files the importer reads were
 * written with. */
#include "h3_quant.h"

#include <math.h>
#include <string.h>

enum { QK = 32, QK_K = 256 };

float h3_half_to_float(uint16_t half) {
    uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t mantissa = half & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (!mantissa) {
            bits = sign;
        } else { /* subnormal: renormalize */
            exponent = 127 - 15 + 1;
            while (!(mantissa & 0x400u)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3ffu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float bf16_to_float(uint16_t bits16) {
    uint32_t bits = (uint32_t)bits16 << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t read_u16(const uint8_t *bytes) {
    return (uint16_t)(bytes[0] | (bytes[1] << 8));
}

static uint32_t read_u32(const uint8_t *bytes) {
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

h3_quant_type h3_quant_from_ggml(uint32_t ggml_type) {
    switch (ggml_type) {
    case 0: return H3_QUANT_F32;
    case 1: return H3_QUANT_F16;
    case 2: return H3_QUANT_Q4_0;
    case 3: return H3_QUANT_Q4_1;
    case 6: return H3_QUANT_Q5_0;
    case 7: return H3_QUANT_Q5_1;
    case 8: return H3_QUANT_Q8_0;
    case 10: return H3_QUANT_Q2_K;
    case 11: return H3_QUANT_Q3_K;
    case 12: return H3_QUANT_Q4_K;
    case 13: return H3_QUANT_Q5_K;
    case 14: return H3_QUANT_Q6_K;
    case 30: return H3_QUANT_BF16;
    default: return H3_QUANT_NONE;
    }
}

const char *h3_quant_name(h3_quant_type type) {
    switch (type) {
    case H3_QUANT_F32: return "F32";
    case H3_QUANT_F16: return "F16";
    case H3_QUANT_BF16: return "BF16";
    case H3_QUANT_Q4_0: return "Q4_0";
    case H3_QUANT_Q4_1: return "Q4_1";
    case H3_QUANT_Q5_0: return "Q5_0";
    case H3_QUANT_Q5_1: return "Q5_1";
    case H3_QUANT_Q8_0: return "Q8_0";
    case H3_QUANT_Q2_K: return "Q2_K";
    case H3_QUANT_Q3_K: return "Q3_K";
    case H3_QUANT_Q4_K: return "Q4_K";
    case H3_QUANT_Q5_K: return "Q5_K";
    case H3_QUANT_Q6_K: return "Q6_K";
    case H3_QUANT_NONE: break;
    }
    return "unsupported";
}

uint32_t h3_quant_block_elements(h3_quant_type type) {
    switch (type) {
    case H3_QUANT_F32: case H3_QUANT_F16: case H3_QUANT_BF16: return 1;
    case H3_QUANT_Q4_0: case H3_QUANT_Q4_1: case H3_QUANT_Q5_0:
    case H3_QUANT_Q5_1: case H3_QUANT_Q8_0: return QK;
    case H3_QUANT_Q2_K: case H3_QUANT_Q3_K: case H3_QUANT_Q4_K:
    case H3_QUANT_Q5_K: case H3_QUANT_Q6_K: return QK_K;
    case H3_QUANT_NONE: break;
    }
    return 0;
}

uint32_t h3_quant_block_bytes(h3_quant_type type) {
    switch (type) {
    case H3_QUANT_F32: return 4;
    case H3_QUANT_F16: case H3_QUANT_BF16: return 2;
    case H3_QUANT_Q4_0: return 2 + QK / 2;                     /* 18 */
    case H3_QUANT_Q4_1: return 4 + QK / 2;                     /* 20 */
    case H3_QUANT_Q5_0: return 2 + 4 + QK / 2;                 /* 22 */
    case H3_QUANT_Q5_1: return 4 + 4 + QK / 2;                 /* 24 */
    case H3_QUANT_Q8_0: return 2 + QK;                         /* 34 */
    case H3_QUANT_Q2_K: return QK_K / 16 + QK_K / 4 + 4;       /* 84 */
    case H3_QUANT_Q3_K: return QK_K / 8 + QK_K / 4 + 12 + 2;   /* 110 */
    case H3_QUANT_Q4_K: return 4 + 12 + QK_K / 2;              /* 144 */
    case H3_QUANT_Q5_K: return 4 + 12 + QK_K / 8 + QK_K / 2;   /* 176 */
    case H3_QUANT_Q6_K: return QK_K / 2 + QK_K / 4 + QK_K / 16 + 2; /* 210 */
    case H3_QUANT_NONE: break;
    }
    return 0;
}

uint64_t h3_quant_bytes(h3_quant_type type, uint64_t count) {
    uint32_t elements = h3_quant_block_elements(type);
    if (!elements || count % elements) return 0;
    return count / elements * h3_quant_block_bytes(type);
}

/* Q4_K / Q5_K: the 6-bit scale and min of sub-block j (ggml's
 * get_scale_min_k4). */
static void scale_min_k4(int j, const uint8_t *q, uint8_t *scale, uint8_t *min) {
    if (j < 4) {
        *scale = q[j] & 63;
        *min = q[j + 4] & 63;
    } else {
        *scale = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *min = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static void block_q4_0(const uint8_t *b, float *y) {
    float d = h3_half_to_float(read_u16(b));
    const uint8_t *qs = b + 2;
    for (int j = 0; j < QK / 2; j++) {
        y[j] = (float)((qs[j] & 0x0F) - 8) * d;
        y[j + QK / 2] = (float)((qs[j] >> 4) - 8) * d;
    }
}

static void block_q4_1(const uint8_t *b, float *y) {
    float d = h3_half_to_float(read_u16(b));
    float m = h3_half_to_float(read_u16(b + 2));
    const uint8_t *qs = b + 4;
    for (int j = 0; j < QK / 2; j++) {
        y[j] = (float)(qs[j] & 0x0F) * d + m;
        y[j + QK / 2] = (float)(qs[j] >> 4) * d + m;
    }
}

static void block_q5_0(const uint8_t *b, float *y) {
    float d = h3_half_to_float(read_u16(b));
    uint32_t qh = read_u32(b + 2);
    const uint8_t *qs = b + 6;
    for (int j = 0; j < QK / 2; j++) {
        uint8_t h0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
        uint8_t h1 = (uint8_t)((qh >> (j + 12)) & 0x10);
        y[j] = (float)(((qs[j] & 0x0F) | h0) - 16) * d;
        y[j + QK / 2] = (float)(((qs[j] >> 4) | h1) - 16) * d;
    }
}

static void block_q5_1(const uint8_t *b, float *y) {
    float d = h3_half_to_float(read_u16(b));
    float m = h3_half_to_float(read_u16(b + 2));
    uint32_t qh = read_u32(b + 4);
    const uint8_t *qs = b + 8;
    for (int j = 0; j < QK / 2; j++) {
        uint8_t h0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
        uint8_t h1 = (uint8_t)((qh >> (j + 12)) & 0x10);
        y[j] = (float)((qs[j] & 0x0F) | h0) * d + m;
        y[j + QK / 2] = (float)((qs[j] >> 4) | h1) * d + m;
    }
}

static void block_q8_0(const uint8_t *b, float *y) {
    float d = h3_half_to_float(read_u16(b));
    const int8_t *qs = (const int8_t *)(b + 2);
    for (int j = 0; j < QK; j++) y[j] = (float)qs[j] * d;
}

static void block_q2_k(const uint8_t *b, float *y) {
    const uint8_t *scales = b;
    const uint8_t *q = b + 16;
    float d = h3_half_to_float(read_u16(b + 80));
    float dmin = h3_half_to_float(read_u16(b + 82));
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc = scales[is++];
            float dl = d * (float)(sc & 0xF), ml = dmin * (float)(sc >> 4);
            for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l] >> shift) & 3) - ml;
            sc = scales[is++];
            dl = d * (float)(sc & 0xF);
            ml = dmin * (float)(sc >> 4);
            for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l + 16] >> shift) & 3) - ml;
            shift += 2;
        }
        q += 32;
    }
}

static void block_q3_k(const uint8_t *b, float *y) {
    const uint8_t *hm = b;
    const uint8_t *q = b + 32;
    const uint8_t *packed = b + 96;
    float d_all = h3_half_to_float(read_u16(b + 108));
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    memcpy(aux, packed, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    int8_t scales[16];
    memcpy(scales, aux, sizeof(scales));
    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            float dl = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; l++)
                *y++ = dl * (float)((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            dl = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; l++)
                *y++ = dl * (float)((int8_t)((q[l + 16] >> shift) & 3) -
                                    ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m = (uint8_t)(m << 1);
        }
        q += 32;
    }
}

static void block_q4_k(const uint8_t *b, float *y) {
    float d = h3_half_to_float(read_u16(b));
    float dmin = h3_half_to_float(read_u16(b + 2));
    const uint8_t *scales = b + 4;
    const uint8_t *q = b + 16;
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        scale_min_k4(is + 0, scales, &sc, &m);
        float d1 = d * (float)sc, m1 = dmin * (float)m;
        scale_min_k4(is + 1, scales, &sc, &m);
        float d2 = d * (float)sc, m2 = dmin * (float)m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (float)(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (float)(q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

static void block_q5_k(const uint8_t *b, float *y) {
    float d = h3_half_to_float(read_u16(b));
    float dmin = h3_half_to_float(read_u16(b + 2));
    const uint8_t *scales = b + 4;
    const uint8_t *qh = b + 16;
    const uint8_t *ql = b + 48;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        scale_min_k4(is + 0, scales, &sc, &m);
        float d1 = d * (float)sc, m1 = dmin * (float)m;
        scale_min_k4(is + 1, scales, &sc, &m);
        float d2 = d * (float)sc, m2 = dmin * (float)m;
        for (int l = 0; l < 32; l++)
            *y++ = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++)
            *y++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 = (uint8_t)(u1 << 2);
        u2 = (uint8_t)(u2 << 2);
    }
}

static void block_q6_k(const uint8_t *b, float *y) {
    const uint8_t *ql = b;
    const uint8_t *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    float d = h3_half_to_float(read_u16(b + 208));
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = ((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = ((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

int h3_quant_dequantize(h3_quant_type type, const void *source, uint64_t count,
                        float *out) {
    uint32_t elements = h3_quant_block_elements(type);
    uint32_t bytes = h3_quant_block_bytes(type);
    if (!elements || !source || !out || count % elements) return 0;
    const uint8_t *b = source;
    uint64_t blocks = count / elements;
    for (uint64_t block = 0; block < blocks; block++, b += bytes) {
        float *y = out + block * elements;
        switch (type) {
        case H3_QUANT_F32: memcpy(y, b, 4); break;
        case H3_QUANT_F16: *y = h3_half_to_float(read_u16(b)); break;
        case H3_QUANT_BF16: *y = bf16_to_float(read_u16(b)); break;
        case H3_QUANT_Q4_0: block_q4_0(b, y); break;
        case H3_QUANT_Q4_1: block_q4_1(b, y); break;
        case H3_QUANT_Q5_0: block_q5_0(b, y); break;
        case H3_QUANT_Q5_1: block_q5_1(b, y); break;
        case H3_QUANT_Q8_0: block_q8_0(b, y); break;
        case H3_QUANT_Q2_K: block_q2_k(b, y); break;
        case H3_QUANT_Q3_K: block_q3_k(b, y); break;
        case H3_QUANT_Q4_K: block_q4_k(b, y); break;
        case H3_QUANT_Q5_K: block_q5_k(b, y); break;
        case H3_QUANT_Q6_K: block_q6_k(b, y); break;
        case H3_QUANT_NONE: return 0;
        }
    }
    return 1;
}

int h3_quant_dequantize_bf16(h3_quant_type type, const void *source,
                             uint64_t count, uint16_t *out) {
    uint32_t elements = h3_quant_block_elements(type);
    uint32_t bytes = h3_quant_block_bytes(type);
    if (!elements || !source || !out || count % elements) return 0;
    float block[QK_K];
    const uint8_t *b = source;
    for (uint64_t first = 0; first < count; first += elements, b += bytes) {
        if (!h3_quant_dequantize(type, b, elements, block)) return 0;
        for (uint32_t j = 0; j < elements; j++) {
            uint32_t bits;
            memcpy(&bits, &block[j], sizeof(bits));
            bits += 0x7fffu + ((bits >> 16) & 1u);
            out[first + j] = (uint16_t)(bits >> 16);
        }
    }
    return 1;
}

int h3_mlx_dequantize_row(const uint32_t *packed, const uint16_t *scales,
                          const uint16_t *biases, uint32_t columns,
                          uint32_t bits, uint32_t group_size, float *out) {
    /* Power-of-two widths pack whole values into each uint32; 3/5/6-bit
     * rows span words and are not read yet. */
    if (!packed || !scales || !biases || !out || !group_size ||
        columns % group_size || (bits != 2 && bits != 4 && bits != 8))
        return 0;
    uint32_t per_word = 32 / bits, mask = (1u << bits) - 1u;
    if (group_size % per_word) return 0;
    for (uint32_t column = 0; column < columns; column++) {
        uint32_t word = packed[column / per_word];
        uint32_t q = (word >> (bits * (column % per_word))) & mask;
        uint32_t group = column / group_size;
        out[column] = (float)q * bf16_to_float(scales[group]) +
                      bf16_to_float(biases[group]);
    }
    return 1;
}
