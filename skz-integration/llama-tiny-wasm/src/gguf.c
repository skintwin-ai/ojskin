/**
 * gguf.c - GGUF file format parser implementation
 * 
 * SKZ Autonomous Agents Integration
 * OJSkin - Open Journal Systems with Skin Zone Journal
 */

#include "../include/gguf.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Error Handling
// ============================================================================

static char gguf_error_msg[256] = {0};

static void gguf_set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(gguf_error_msg, sizeof(gguf_error_msg), fmt, args);
    va_end(args);
}

const char* gguf_get_error(void) {
    return gguf_error_msg[0] ? gguf_error_msg : NULL;
}

// ============================================================================
// Type Information
// ============================================================================

static const size_t GGML_TYPE_SIZES[] = {
    [GGML_TYPE_F32]  = sizeof(float),
    [GGML_TYPE_F16]  = sizeof(uint16_t),
    [GGML_TYPE_Q4_0] = sizeof(uint16_t) + 16,  // scale + 32 4-bit values
    [GGML_TYPE_Q4_1] = sizeof(uint16_t) * 2 + 16,
    [GGML_TYPE_Q5_0] = sizeof(uint16_t) + 4 + 16,
    [GGML_TYPE_Q5_1] = sizeof(uint16_t) * 2 + 4 + 16,
    [GGML_TYPE_Q8_0] = sizeof(uint16_t) + 32,
    [GGML_TYPE_Q8_1] = sizeof(float) * 2 + 32,
    [GGML_TYPE_Q2_K] = 256 / 16 + 256 / 4 + 2 + 2,
    [GGML_TYPE_Q3_K] = 256 / 8 + 256 / 4 + 12 + 2,
    [GGML_TYPE_Q4_K] = 2 + 2 + 12 + 256 / 2,
    [GGML_TYPE_Q5_K] = 2 + 2 + 12 + 256 / 8 + 256 / 2,
    [GGML_TYPE_Q6_K] = 256 / 2 + 256 / 4 + 256 / 16 + 2,
    [GGML_TYPE_Q8_K] = sizeof(float) + 256 + 16,
    [GGML_TYPE_I8]   = sizeof(int8_t),
    [GGML_TYPE_I16]  = sizeof(int16_t),
    [GGML_TYPE_I32]  = sizeof(int32_t),
};

static const int GGML_BLCK_SIZES[] = {
    [GGML_TYPE_F32]  = 1,
    [GGML_TYPE_F16]  = 1,
    [GGML_TYPE_Q4_0] = 32,
    [GGML_TYPE_Q4_1] = 32,
    [GGML_TYPE_Q5_0] = 32,
    [GGML_TYPE_Q5_1] = 32,
    [GGML_TYPE_Q8_0] = 32,
    [GGML_TYPE_Q8_1] = 32,
    [GGML_TYPE_Q2_K] = 256,
    [GGML_TYPE_Q3_K] = 256,
    [GGML_TYPE_Q4_K] = 256,
    [GGML_TYPE_Q5_K] = 256,
    [GGML_TYPE_Q6_K] = 256,
    [GGML_TYPE_Q8_K] = 256,
    [GGML_TYPE_I8]   = 1,
    [GGML_TYPE_I16]  = 1,
    [GGML_TYPE_I32]  = 1,
};

static const char* GGML_TYPE_NAMES[] = {
    [GGML_TYPE_F32]  = "f32",
    [GGML_TYPE_F16]  = "f16",
    [GGML_TYPE_Q4_0] = "q4_0",
    [GGML_TYPE_Q4_1] = "q4_1",
    [GGML_TYPE_Q5_0] = "q5_0",
    [GGML_TYPE_Q5_1] = "q5_1",
    [GGML_TYPE_Q8_0] = "q8_0",
    [GGML_TYPE_Q8_1] = "q8_1",
    [GGML_TYPE_Q2_K] = "q2_k",
    [GGML_TYPE_Q3_K] = "q3_k",
    [GGML_TYPE_Q4_K] = "q4_k",
    [GGML_TYPE_Q5_K] = "q5_k",
    [GGML_TYPE_Q6_K] = "q6_k",
    [GGML_TYPE_Q8_K] = "q8_k",
    [GGML_TYPE_I8]   = "i8",
    [GGML_TYPE_I16]  = "i16",
    [GGML_TYPE_I32]  = "i32",
};

size_t ggml_type_size(ggml_type type) {
    if (type >= GGML_TYPE_COUNT) return 0;
    return GGML_TYPE_SIZES[type];
}

int ggml_blck_size(ggml_type type) {
    if (type >= GGML_TYPE_COUNT) return 1;
    return GGML_BLCK_SIZES[type];
}

bool ggml_is_quantized(ggml_type type) {
    return type >= GGML_TYPE_Q4_0 && type <= GGML_TYPE_Q8_K;
}

const char* ggml_type_name(ggml_type type) {
    if (type >= GGML_TYPE_COUNT) return "unknown";
    return GGML_TYPE_NAMES[type];
}

// ============================================================================
// Buffer Reader
// ============================================================================

typedef struct {
    const uint8_t* data;
    size_t size;
    size_t pos;
} buffer_reader;

static bool reader_init(buffer_reader* r, const void* data, size_t size) {
    r->data = (const uint8_t*)data;
    r->size = size;
    r->pos = 0;
    return true;
}

static bool reader_can_read(const buffer_reader* r, size_t n) {
    return r->pos + n <= r->size;
}

static bool reader_read(buffer_reader* r, void* dst, size_t n) {
    if (!reader_can_read(r, n)) {
        gguf_set_error("Unexpected end of file at offset %zu", r->pos);
        return false;
    }
    memcpy(dst, r->data + r->pos, n);
    r->pos += n;
    return true;
}

static bool reader_skip(buffer_reader* r, size_t n) {
    if (!reader_can_read(r, n)) {
        gguf_set_error("Unexpected end of file at offset %zu", r->pos);
        return false;
    }
    r->pos += n;
    return true;
}

static uint8_t reader_read_u8(buffer_reader* r) {
    uint8_t v = 0;
    reader_read(r, &v, 1);
    return v;
}

static uint32_t reader_read_u32(buffer_reader* r) {
    uint32_t v = 0;
    reader_read(r, &v, 4);
    return v;
}

static uint64_t reader_read_u64(buffer_reader* r) {
    uint64_t v = 0;
    reader_read(r, &v, 8);
    return v;
}

static int32_t reader_read_i32(buffer_reader* r) {
    int32_t v = 0;
    reader_read(r, &v, 4);
    return v;
}

static int64_t reader_read_i64(buffer_reader* r) {
    int64_t v = 0;
    reader_read(r, &v, 8);
    return v;
}

static float reader_read_f32(buffer_reader* r) {
    float v = 0;
    reader_read(r, &v, 4);
    return v;
}

static double reader_read_f64(buffer_reader* r) {
    double v = 0;
    reader_read(r, &v, 8);
    return v;
}

static char* reader_read_string(buffer_reader* r) {
    uint64_t len = reader_read_u64(r);
    if (len > GGUF_MAX_KEY_LENGTH * 10) {
        gguf_set_error("String too long: %llu", (unsigned long long)len);
        return NULL;
    }
    
    char* str = (char*)malloc(len + 1);
    if (!str) {
        gguf_set_error("Out of memory for string");
        return NULL;
    }
    
    if (!reader_read(r, str, len)) {
        free(str);
        return NULL;
    }
    str[len] = '\0';
    return str;
}

// ============================================================================
// GGUF Parsing
// ============================================================================

static bool parse_kv_value(buffer_reader* r, gguf_kv* kv) {
    kv->type = (gguf_type)reader_read_u32(r);
    
    switch (kv->type) {
        case GGUF_TYPE_UINT8:
            kv->value.u8 = reader_read_u8(r);
            break;
        case GGUF_TYPE_INT8:
            kv->value.i8 = (int8_t)reader_read_u8(r);
            break;
        case GGUF_TYPE_UINT16:
            reader_read(r, &kv->value.u16, 2);
            break;
        case GGUF_TYPE_INT16:
            reader_read(r, &kv->value.i16, 2);
            break;
        case GGUF_TYPE_UINT32:
            kv->value.u32 = reader_read_u32(r);
            break;
        case GGUF_TYPE_INT32:
            kv->value.i32 = reader_read_i32(r);
            break;
        case GGUF_TYPE_UINT64:
            kv->value.u64 = reader_read_u64(r);
            break;
        case GGUF_TYPE_INT64:
            kv->value.i64 = reader_read_i64(r);
            break;
        case GGUF_TYPE_FLOAT32:
            kv->value.f32 = reader_read_f32(r);
            break;
        case GGUF_TYPE_FLOAT64:
            kv->value.f64 = reader_read_f64(r);
            break;
        case GGUF_TYPE_BOOL:
            kv->value.b = reader_read_u8(r) != 0;
            break;
        case GGUF_TYPE_STRING:
            kv->value.str = reader_read_string(r);
            if (!kv->value.str) return false;
            break;
        case GGUF_TYPE_ARRAY: {
            kv->value.arr.elem_type = (gguf_type)reader_read_u32(r);
            kv->value.arr.n_elem = reader_read_u64(r);
            
            // For arrays, we store the raw data pointer
            size_t elem_size = 0;
            switch (kv->value.arr.elem_type) {
                case GGUF_TYPE_UINT8:
                case GGUF_TYPE_INT8:
                case GGUF_TYPE_BOOL:
                    elem_size = 1;
                    break;
                case GGUF_TYPE_UINT16:
                case GGUF_TYPE_INT16:
                    elem_size = 2;
                    break;
                case GGUF_TYPE_UINT32:
                case GGUF_TYPE_INT32:
                case GGUF_TYPE_FLOAT32:
                    elem_size = 4;
                    break;
                case GGUF_TYPE_UINT64:
                case GGUF_TYPE_INT64:
                case GGUF_TYPE_FLOAT64:
                    elem_size = 8;
                    break;
                case GGUF_TYPE_STRING: {
                    // Array of strings - allocate and read each
                    char** strings = (char**)malloc(kv->value.arr.n_elem * sizeof(char*));
                    if (!strings) {
                        gguf_set_error("Out of memory for string array");
                        return false;
                    }
                    for (uint64_t i = 0; i < kv->value.arr.n_elem; i++) {
                        strings[i] = reader_read_string(r);
                        if (!strings[i]) {
                            for (uint64_t j = 0; j < i; j++) free(strings[j]);
                            free(strings);
                            return false;
                        }
                    }
                    kv->value.arr.data = strings;
                    return true;
                }
                default:
                    gguf_set_error("Unsupported array element type: %d", kv->value.arr.elem_type);
                    return false;
            }
            
            size_t total_size = elem_size * kv->value.arr.n_elem;
            void* data = malloc(total_size);
            if (!data) {
                gguf_set_error("Out of memory for array");
                return false;
            }
            if (!reader_read(r, data, total_size)) {
                free(data);
                return false;
            }
            kv->value.arr.data = data;
            break;
        }
        default:
            gguf_set_error("Unknown KV type: %d", kv->type);
            return false;
    }
    
    return true;
}

gguf_context* gguf_parse(const void* data, size_t size) {
    buffer_reader r;
    reader_init(&r, data, size);
    
    // Allocate context
    gguf_context* ctx = (gguf_context*)calloc(1, sizeof(gguf_context));
    if (!ctx) {
        gguf_set_error("Out of memory for GGUF context");
        return NULL;
    }
    
    ctx->data = data;
    ctx->data_size = size;
    
    // Read header
    if (!reader_read(&r, &ctx->header, sizeof(gguf_header))) {
        free(ctx);
        return NULL;
    }
    
    // Validate magic
    if (memcmp(ctx->header.magic, "GGUF", 4) != 0) {
        gguf_set_error("Invalid GGUF magic: expected 'GGUF'");
        free(ctx);
        return NULL;
    }
    
    // Validate version
    if (ctx->header.version < 2 || ctx->header.version > 3) {
        gguf_set_error("Unsupported GGUF version: %d (expected 2 or 3)", ctx->header.version);
        free(ctx);
        return NULL;
    }
    
    ctx->n_kv = ctx->header.n_kv;
    ctx->n_tensors = ctx->header.n_tensors;
    
    // Allocate KV pairs
    if (ctx->n_kv > 0) {
        ctx->kv = (gguf_kv*)calloc(ctx->n_kv, sizeof(gguf_kv));
        if (!ctx->kv) {
            gguf_set_error("Out of memory for KV pairs");
            gguf_free(ctx);
            return NULL;
        }
        
        // Read KV pairs
        for (uint64_t i = 0; i < ctx->n_kv; i++) {
            ctx->kv[i].key = reader_read_string(&r);
            if (!ctx->kv[i].key) {
                gguf_free(ctx);
                return NULL;
            }
            
            if (!parse_kv_value(&r, &ctx->kv[i])) {
                gguf_free(ctx);
                return NULL;
            }
        }
    }
    
    // Allocate tensors
    if (ctx->n_tensors > 0) {
        ctx->tensors = (gguf_tensor_info*)calloc(ctx->n_tensors, sizeof(gguf_tensor_info));
        if (!ctx->tensors) {
            gguf_set_error("Out of memory for tensors");
            gguf_free(ctx);
            return NULL;
        }
        
        // Read tensor info
        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            ctx->tensors[i].name = reader_read_string(&r);
            if (!ctx->tensors[i].name) {
                gguf_free(ctx);
                return NULL;
            }
            
            ctx->tensors[i].n_dims = reader_read_u32(&r);
            if (ctx->tensors[i].n_dims > 4) {
                gguf_set_error("Too many dimensions: %d", ctx->tensors[i].n_dims);
                gguf_free(ctx);
                return NULL;
            }
            
            for (uint32_t j = 0; j < ctx->tensors[i].n_dims; j++) {
                ctx->tensors[i].ne[j] = reader_read_u64(&r);
            }
            for (uint32_t j = ctx->tensors[i].n_dims; j < 4; j++) {
                ctx->tensors[i].ne[j] = 1;
            }
            
            ctx->tensors[i].type = (ggml_type)reader_read_u32(&r);
            ctx->tensors[i].offset = reader_read_u64(&r);
            
            // Calculate tensor size
            uint64_t n_elem = gguf_tensor_n_elements(&ctx->tensors[i]);
            int blck_size = ggml_blck_size(ctx->tensors[i].type);
            size_t type_size = ggml_type_size(ctx->tensors[i].type);
            ctx->tensors[i].size = (n_elem / blck_size) * type_size;
        }
    }
    
    // Align to 32 bytes for tensor data
    size_t alignment = 32;
    ctx->data_offset = ((r.pos + alignment - 1) / alignment) * alignment;
    
    return ctx;
}

void gguf_free(gguf_context* ctx) {
    if (!ctx) return;
    
    // Free KV pairs
    if (ctx->kv) {
        for (uint64_t i = 0; i < ctx->n_kv; i++) {
            free(ctx->kv[i].key);
            if (ctx->kv[i].type == GGUF_TYPE_STRING) {
                free(ctx->kv[i].value.str);
            } else if (ctx->kv[i].type == GGUF_TYPE_ARRAY) {
                if (ctx->kv[i].value.arr.elem_type == GGUF_TYPE_STRING) {
                    char** strings = (char**)ctx->kv[i].value.arr.data;
                    for (uint64_t j = 0; j < ctx->kv[i].value.arr.n_elem; j++) {
                        free(strings[j]);
                    }
                }
                free(ctx->kv[i].value.arr.data);
            }
        }
        free(ctx->kv);
    }
    
    // Free tensors
    if (ctx->tensors) {
        for (uint64_t i = 0; i < ctx->n_tensors; i++) {
            free(ctx->tensors[i].name);
        }
        free(ctx->tensors);
    }
    
    free(ctx);
}

// ============================================================================
// Key-Value Access
// ============================================================================

const gguf_kv* gguf_find_kv(const gguf_context* ctx, const char* key) {
    if (!ctx || !key) return NULL;
    
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kv[i].key, key) == 0) {
            return &ctx->kv[i];
        }
    }
    return NULL;
}

const char* gguf_get_str(const gguf_context* ctx, const char* key, const char* default_value) {
    const gguf_kv* kv = gguf_find_kv(ctx, key);
    if (!kv || kv->type != GGUF_TYPE_STRING) return default_value;
    return kv->value.str;
}

int64_t gguf_get_int(const gguf_context* ctx, const char* key, int64_t default_value) {
    const gguf_kv* kv = gguf_find_kv(ctx, key);
    if (!kv) return default_value;
    
    switch (kv->type) {
        case GGUF_TYPE_INT8:   return kv->value.i8;
        case GGUF_TYPE_INT16:  return kv->value.i16;
        case GGUF_TYPE_INT32:  return kv->value.i32;
        case GGUF_TYPE_INT64:  return kv->value.i64;
        case GGUF_TYPE_UINT8:  return kv->value.u8;
        case GGUF_TYPE_UINT16: return kv->value.u16;
        case GGUF_TYPE_UINT32: return kv->value.u32;
        case GGUF_TYPE_UINT64: return (int64_t)kv->value.u64;
        default: return default_value;
    }
}

uint64_t gguf_get_uint(const gguf_context* ctx, const char* key, uint64_t default_value) {
    const gguf_kv* kv = gguf_find_kv(ctx, key);
    if (!kv) return default_value;
    
    switch (kv->type) {
        case GGUF_TYPE_UINT8:  return kv->value.u8;
        case GGUF_TYPE_UINT16: return kv->value.u16;
        case GGUF_TYPE_UINT32: return kv->value.u32;
        case GGUF_TYPE_UINT64: return kv->value.u64;
        case GGUF_TYPE_INT8:   return (uint64_t)kv->value.i8;
        case GGUF_TYPE_INT16:  return (uint64_t)kv->value.i16;
        case GGUF_TYPE_INT32:  return (uint64_t)kv->value.i32;
        case GGUF_TYPE_INT64:  return (uint64_t)kv->value.i64;
        default: return default_value;
    }
}

float gguf_get_float(const gguf_context* ctx, const char* key, float default_value) {
    const gguf_kv* kv = gguf_find_kv(ctx, key);
    if (!kv) return default_value;
    
    switch (kv->type) {
        case GGUF_TYPE_FLOAT32: return kv->value.f32;
        case GGUF_TYPE_FLOAT64: return (float)kv->value.f64;
        default: return default_value;
    }
}

bool gguf_get_bool(const gguf_context* ctx, const char* key, bool default_value) {
    const gguf_kv* kv = gguf_find_kv(ctx, key);
    if (!kv || kv->type != GGUF_TYPE_BOOL) return default_value;
    return kv->value.b;
}

const void* gguf_get_array(const gguf_context* ctx, const char* key, uint64_t* n_elem, gguf_type* elem_type) {
    const gguf_kv* kv = gguf_find_kv(ctx, key);
    if (!kv || kv->type != GGUF_TYPE_ARRAY) return NULL;
    
    if (n_elem) *n_elem = kv->value.arr.n_elem;
    if (elem_type) *elem_type = kv->value.arr.elem_type;
    return kv->value.arr.data;
}

// ============================================================================
// Tensor Access
// ============================================================================

const gguf_tensor_info* gguf_find_tensor(const gguf_context* ctx, const char* name) {
    if (!ctx || !name) return NULL;
    
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

const void* gguf_get_tensor_data(const gguf_context* ctx, const gguf_tensor_info* tensor) {
    if (!ctx || !tensor) return NULL;
    return (const uint8_t*)ctx->data + ctx->data_offset + tensor->offset;
}

size_t gguf_tensor_size(const gguf_tensor_info* tensor) {
    if (!tensor) return 0;
    return tensor->size;
}

uint64_t gguf_tensor_n_elements(const gguf_tensor_info* tensor) {
    if (!tensor) return 0;
    return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}
