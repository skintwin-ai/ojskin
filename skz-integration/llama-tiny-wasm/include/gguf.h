/**
 * gguf.h - Minimal GGUF file format parser
 * 
 * SKZ Autonomous Agents Integration
 * OJSkin - Open Journal Systems with Skin Zone Journal
 * 
 * Lightweight GGUF parser for loading quantized LLM models.
 * Supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 quantization types.
 */

#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// GGUF Constants
// ============================================================================

#define GGUF_MAGIC 0x46554747  // "GGUF" in little-endian
#define GGUF_VERSION 3

// Maximum sizes
#define GGUF_MAX_KEY_LENGTH 256
#define GGUF_MAX_TENSORS 1024
#define GGUF_MAX_KV_PAIRS 512

// ============================================================================
// GGUF Types
// ============================================================================

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type;

typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_I8      = 16,
    GGML_TYPE_I16     = 17,
    GGML_TYPE_I32     = 18,
    GGML_TYPE_COUNT,
} ggml_type;

// ============================================================================
// GGUF Structures
// ============================================================================

typedef struct {
    char magic[4];         // "GGUF"
    uint32_t version;      // File format version (3)
    uint64_t n_tensors;    // Number of tensors
    uint64_t n_kv;         // Number of key-value pairs
} gguf_header;

typedef struct {
    char* key;
    gguf_type type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        uint64_t u64;
        int64_t  i64;
        float    f32;
        double   f64;
        bool     b;
        char*    str;
        struct {
            gguf_type elem_type;
            uint64_t  n_elem;
            void*     data;
        } arr;
    } value;
} gguf_kv;

typedef struct {
    char*    name;
    uint32_t n_dims;
    uint64_t ne[4];        // Dimensions
    ggml_type type;
    uint64_t offset;       // Offset in data section
    size_t   size;         // Size in bytes
} gguf_tensor_info;

typedef struct {
    gguf_header header;
    gguf_kv* kv;
    uint64_t n_kv;
    gguf_tensor_info* tensors;
    uint64_t n_tensors;
    size_t data_offset;    // Offset to tensor data
    const void* data;      // Pointer to raw file data
    size_t data_size;      // Total file size
} gguf_context;

// ============================================================================
// GGUF Parsing
// ============================================================================

/**
 * Parse GGUF file from memory buffer.
 * 
 * @param data    Pointer to GGUF file data
 * @param size    Size of data in bytes
 * @return        GGUF context, or NULL on error
 */
gguf_context* gguf_parse(const void* data, size_t size);

/**
 * Free GGUF context.
 */
void gguf_free(gguf_context* ctx);

/**
 * Get last error message.
 */
const char* gguf_get_error(void);

// ============================================================================
// Key-Value Access
// ============================================================================

/**
 * Find key-value pair by key.
 * Returns NULL if not found.
 */
const gguf_kv* gguf_find_kv(const gguf_context* ctx, const char* key);

/**
 * Get string value by key.
 * Returns default_value if not found or wrong type.
 */
const char* gguf_get_str(const gguf_context* ctx, const char* key, const char* default_value);

/**
 * Get integer value by key.
 */
int64_t gguf_get_int(const gguf_context* ctx, const char* key, int64_t default_value);

/**
 * Get unsigned integer value by key.
 */
uint64_t gguf_get_uint(const gguf_context* ctx, const char* key, uint64_t default_value);

/**
 * Get float value by key.
 */
float gguf_get_float(const gguf_context* ctx, const char* key, float default_value);

/**
 * Get boolean value by key.
 */
bool gguf_get_bool(const gguf_context* ctx, const char* key, bool default_value);

/**
 * Get array value by key.
 * Returns NULL if not found or wrong type.
 */
const void* gguf_get_array(const gguf_context* ctx, const char* key, uint64_t* n_elem, gguf_type* elem_type);

// ============================================================================
// Tensor Access
// ============================================================================

/**
 * Find tensor by name.
 * Returns NULL if not found.
 */
const gguf_tensor_info* gguf_find_tensor(const gguf_context* ctx, const char* name);

/**
 * Get pointer to tensor data.
 */
const void* gguf_get_tensor_data(const gguf_context* ctx, const gguf_tensor_info* tensor);

/**
 * Calculate tensor size in bytes.
 */
size_t gguf_tensor_size(const gguf_tensor_info* tensor);

/**
 * Get number of elements in tensor.
 */
uint64_t gguf_tensor_n_elements(const gguf_tensor_info* tensor);

// ============================================================================
// Type Information
// ============================================================================

/**
 * Get size of GGML type in bytes (per element or per block).
 */
size_t ggml_type_size(ggml_type type);

/**
 * Get block size for quantized types.
 */
int ggml_blck_size(ggml_type type);

/**
 * Check if type is quantized.
 */
bool ggml_is_quantized(ggml_type type);

/**
 * Get type name string.
 */
const char* ggml_type_name(ggml_type type);

// ============================================================================
// Model Hyperparameters (Common Keys)
// ============================================================================

// Architecture
#define GGUF_KEY_GENERAL_ARCHITECTURE       "general.architecture"
#define GGUF_KEY_GENERAL_NAME               "general.name"
#define GGUF_KEY_GENERAL_FILE_TYPE          "general.file_type"

// Model dimensions (replace {arch} with architecture name)
#define GGUF_KEY_CONTEXT_LENGTH             ".context_length"
#define GGUF_KEY_EMBEDDING_LENGTH           ".embedding_length"
#define GGUF_KEY_BLOCK_COUNT                ".block_count"
#define GGUF_KEY_FEED_FORWARD_LENGTH        ".feed_forward_length"
#define GGUF_KEY_ATTENTION_HEAD_COUNT       ".attention.head_count"
#define GGUF_KEY_ATTENTION_HEAD_COUNT_KV    ".attention.head_count_kv"
#define GGUF_KEY_ROPE_FREQ_BASE             ".rope.freq_base"
#define GGUF_KEY_ROPE_SCALE_LINEAR          ".rope.scale_linear"

// Tokenizer
#define GGUF_KEY_TOKENIZER_MODEL            "tokenizer.ggml.model"
#define GGUF_KEY_TOKENIZER_TOKENS           "tokenizer.ggml.tokens"
#define GGUF_KEY_TOKENIZER_SCORES           "tokenizer.ggml.scores"
#define GGUF_KEY_TOKENIZER_TOKEN_TYPE       "tokenizer.ggml.token_type"
#define GGUF_KEY_TOKENIZER_MERGES           "tokenizer.ggml.merges"
#define GGUF_KEY_TOKENIZER_BOS_ID           "tokenizer.ggml.bos_token_id"
#define GGUF_KEY_TOKENIZER_EOS_ID           "tokenizer.ggml.eos_token_id"
#define GGUF_KEY_TOKENIZER_PAD_ID           "tokenizer.ggml.padding_token_id"
#define GGUF_KEY_TOKENIZER_UNK_ID           "tokenizer.ggml.unknown_token_id"

#ifdef __cplusplus
}
#endif

#endif // GGUF_H
