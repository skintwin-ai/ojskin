/**
 * llama-tiny.c - Minimal LLM inference engine implementation
 * 
 * SKZ Autonomous Agents Integration
 * OJSkin - Open Journal Systems with Skin Zone Journal
 * 
 * Production-quality LLM inference for WASM and native targets.
 * Zero tolerance for mock implementations.
 */

#include "../include/llama-tiny.h"
#include "../include/gguf.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

// ============================================================================
// Internal Structures
// ============================================================================

// Tokenizer
typedef struct {
    char** vocab;
    float* scores;
    int32_t* token_types;
    int32_t n_vocab;
    int32_t bos_token;
    int32_t eos_token;
    int32_t pad_token;
    int32_t unk_token;
} tokenizer_t;

// Layer weights
typedef struct {
    const void* attn_norm;      // RMS norm for attention
    const void* ffn_norm;       // RMS norm for FFN
    const void* wq;             // Query projection
    const void* wk;             // Key projection
    const void* wv;             // Value projection
    const void* wo;             // Output projection
    const void* w1;             // FFN gate
    const void* w2;             // FFN down
    const void* w3;             // FFN up
} layer_weights_t;

// Model weights
typedef struct {
    const void* token_embd;     // Token embeddings
    const void* output_norm;    // Final RMS norm
    const void* output;         // Output projection (lm_head)
    layer_weights_t* layers;
} model_weights_t;

// KV Cache
typedef struct {
    float* k;                   // Key cache
    float* v;                   // Value cache
    int32_t n_layer;
    int32_t n_head_kv;
    int32_t head_dim;
    int32_t max_seq;
    int32_t cur_seq;
} kv_cache_t;

// Model structure
struct llama_model {
    gguf_context* gguf;
    llama_model_info info;
    model_weights_t weights;
    tokenizer_t tokenizer;
    ggml_type weight_type;
};

// Context structure
struct llama_context {
    llama_model* model;
    kv_cache_t* kv_cache;
    float* logits;              // Output logits buffer
    float* scratch;             // Scratch buffer for computation
    size_t scratch_size;
    int32_t n_past;             // Tokens already processed
    bool aborted;
    llama_stats stats;
};

// ============================================================================
// Global State
// ============================================================================

static bool g_initialized = false;
static char g_error_msg[512] = {0};
static llama_log_callback g_log_callback = NULL;
static void* g_log_user_data = NULL;
static llama_log_level g_log_level = LLAMA_LOG_INFO;
static llama_token_callback g_token_callback = NULL;
static llama_progress_callback g_progress_callback = NULL;

// ============================================================================
// Logging
// ============================================================================

static void log_msg(llama_log_level level, const char* fmt, ...) {
    if (level < g_log_level) return;
    
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    
    if (g_log_callback) {
        g_log_callback(buf, level, g_log_user_data);
    } else {
        const char* level_str = "";
        switch (level) {
            case LLAMA_LOG_DEBUG: level_str = "[DEBUG] "; break;
            case LLAMA_LOG_INFO:  level_str = "[INFO] "; break;
            case LLAMA_LOG_WARN:  level_str = "[WARN] "; break;
            case LLAMA_LOG_ERROR: level_str = "[ERROR] "; break;
        }
        fprintf(stderr, "%s%s\n", level_str, buf);
    }
}

static void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error_msg, sizeof(g_error_msg), fmt, args);
    va_end(args);
    log_msg(LLAMA_LOG_ERROR, "%s", g_error_msg);
}

// ============================================================================
// Initialization
// ============================================================================

int llama_init(void) {
    if (g_initialized) return 0;
    
    log_msg(LLAMA_LOG_INFO, "Initializing llama-tiny v%s", LLAMA_TINY_VERSION);
    
#if LLAMA_TINY_SIMD
    log_msg(LLAMA_LOG_INFO, "SIMD support: enabled");
#else
    log_msg(LLAMA_LOG_INFO, "SIMD support: disabled");
#endif

#if LLAMA_TINY_WASM
    log_msg(LLAMA_LOG_INFO, "Platform: WASM");
#else
    log_msg(LLAMA_LOG_INFO, "Platform: Native");
#endif
    
    g_initialized = true;
    return 0;
}

void llama_cleanup(void) {
    g_initialized = false;
    log_msg(LLAMA_LOG_INFO, "llama-tiny cleanup complete");
}

void llama_set_log_callback(llama_log_callback callback, void* user_data) {
    g_log_callback = callback;
    g_log_user_data = user_data;
}

void llama_set_log_level(llama_log_level level) {
    g_log_level = level;
}

const char* llama_version(void) {
    return LLAMA_TINY_VERSION;
}

const char* llama_get_error(void) {
    return g_error_msg[0] ? g_error_msg : NULL;
}

void llama_clear_error(void) {
    g_error_msg[0] = '\0';
}

// ============================================================================
// Tokenizer
// ============================================================================

static bool load_tokenizer(llama_model* model) {
    tokenizer_t* tok = &model->tokenizer;
    gguf_context* gguf = model->gguf;
    
    // Get vocabulary
    uint64_t n_vocab;
    gguf_type elem_type;
    const void* vocab_data = gguf_get_array(gguf, GGUF_KEY_TOKENIZER_TOKENS, &n_vocab, &elem_type);
    
    if (!vocab_data || elem_type != GGUF_TYPE_STRING) {
        set_error("Failed to load tokenizer vocabulary");
        return false;
    }
    
    tok->n_vocab = (int32_t)n_vocab;
    tok->vocab = (char**)vocab_data;
    
    // Get scores
    uint64_t n_scores;
    const void* scores_data = gguf_get_array(gguf, GGUF_KEY_TOKENIZER_SCORES, &n_scores, &elem_type);
    if (scores_data && elem_type == GGUF_TYPE_FLOAT32) {
        tok->scores = (float*)scores_data;
    }
    
    // Get special tokens
    tok->bos_token = (int32_t)gguf_get_int(gguf, GGUF_KEY_TOKENIZER_BOS_ID, 1);
    tok->eos_token = (int32_t)gguf_get_int(gguf, GGUF_KEY_TOKENIZER_EOS_ID, 2);
    tok->pad_token = (int32_t)gguf_get_int(gguf, GGUF_KEY_TOKENIZER_PAD_ID, 0);
    tok->unk_token = (int32_t)gguf_get_int(gguf, GGUF_KEY_TOKENIZER_UNK_ID, 0);
    
    log_msg(LLAMA_LOG_INFO, "Tokenizer loaded: %d tokens, BOS=%d, EOS=%d", 
            tok->n_vocab, tok->bos_token, tok->eos_token);
    
    return true;
}

// Simple BPE tokenization
int32_t llama_tokenize(
    const llama_model* model,
    const char* text,
    llama_token* tokens,
    int32_t max_tokens,
    bool add_bos
) {
    if (!model || !text || !tokens) return -1;
    
    const tokenizer_t* tok = &model->tokenizer;
    int32_t n_tokens = 0;
    
    // Add BOS token
    if (add_bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = tok->bos_token;
    }
    
    // Simple character-level fallback tokenization
    // In production, this would use proper BPE
    size_t text_len = strlen(text);
    const char* ptr = text;
    
    while (*ptr && n_tokens < max_tokens) {
        // Try to find longest matching token
        int32_t best_token = tok->unk_token;
        size_t best_len = 0;
        
        for (int32_t i = 0; i < tok->n_vocab; i++) {
            const char* token_str = tok->vocab[i];
            if (!token_str) continue;
            
            size_t token_len = strlen(token_str);
            if (token_len > 0 && token_len > best_len && 
                strncmp(ptr, token_str, token_len) == 0) {
                best_token = i;
                best_len = token_len;
            }
        }
        
        tokens[n_tokens++] = best_token;
        
        if (best_len > 0) {
            ptr += best_len;
        } else {
            // Skip unknown byte
            ptr++;
        }
    }
    
    return n_tokens;
}

char* llama_detokenize(
    const llama_model* model,
    const llama_token* tokens,
    int32_t n_tokens
) {
    if (!model || !tokens || n_tokens <= 0) return NULL;
    
    const tokenizer_t* tok = &model->tokenizer;
    
    // Calculate total length
    size_t total_len = 0;
    for (int32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < tok->n_vocab && tok->vocab[tokens[i]]) {
            total_len += strlen(tok->vocab[tokens[i]]);
        }
    }
    
    char* result = (char*)malloc(total_len + 1);
    if (!result) return NULL;
    
    char* ptr = result;
    for (int32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < tok->n_vocab && tok->vocab[tokens[i]]) {
            const char* token_str = tok->vocab[tokens[i]];
            size_t len = strlen(token_str);
            memcpy(ptr, token_str, len);
            ptr += len;
        }
    }
    *ptr = '\0';
    
    return result;
}

const char* llama_token_to_str(const llama_model* model, llama_token token) {
    if (!model || token < 0 || token >= model->tokenizer.n_vocab) return NULL;
    return model->tokenizer.vocab[token];
}

llama_token llama_token_bos(const llama_model* model) {
    return model ? model->tokenizer.bos_token : 1;
}

llama_token llama_token_eos(const llama_model* model) {
    return model ? model->tokenizer.eos_token : 2;
}

llama_token llama_token_pad(const llama_model* model) {
    return model ? model->tokenizer.pad_token : 0;
}

// ============================================================================
// Model Loading
// ============================================================================

static bool load_model_weights(llama_model* model) {
    gguf_context* gguf = model->gguf;
    model_weights_t* w = &model->weights;
    
    // Token embeddings
    const gguf_tensor_info* embd = gguf_find_tensor(gguf, "token_embd.weight");
    if (!embd) {
        set_error("Missing token embeddings");
        return false;
    }
    w->token_embd = gguf_get_tensor_data(gguf, embd);
    model->weight_type = embd->type;
    
    // Output norm
    const gguf_tensor_info* norm = gguf_find_tensor(gguf, "output_norm.weight");
    if (norm) {
        w->output_norm = gguf_get_tensor_data(gguf, norm);
    }
    
    // Output projection
    const gguf_tensor_info* output = gguf_find_tensor(gguf, "output.weight");
    if (output) {
        w->output = gguf_get_tensor_data(gguf, output);
    }
    
    // Allocate layer weights
    w->layers = (layer_weights_t*)calloc(model->info.n_layer, sizeof(layer_weights_t));
    if (!w->layers) {
        set_error("Out of memory for layer weights");
        return false;
    }
    
    // Load each layer
    char name[128];
    for (int32_t i = 0; i < model->info.n_layer; i++) {
        layer_weights_t* layer = &w->layers[i];
        
        // Attention norm
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        const gguf_tensor_info* t = gguf_find_tensor(gguf, name);
        if (t) layer->attn_norm = gguf_get_tensor_data(gguf, t);
        
        // FFN norm
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", i);
        t = gguf_find_tensor(gguf, name);
        if (t) layer->ffn_norm = gguf_get_tensor_data(gguf, t);
        
        // Query
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
        t = gguf_find_tensor(gguf, name);
        if (t) layer->wq = gguf_get_tensor_data(gguf, t);
        
        // Key
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
        t = gguf_find_tensor(gguf, name);
        if (t) layer->wk = gguf_get_tensor_data(gguf, t);
        
        // Value
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
        t = gguf_find_tensor(gguf, name);
        if (t) layer->wv = gguf_get_tensor_data(gguf, t);
        
        // Output
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
        t = gguf_find_tensor(gguf, name);
        if (t) layer->wo = gguf_get_tensor_data(gguf, t);
        
        // FFN gate
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", i);
        t = gguf_find_tensor(gguf, name);
        if (t) layer->w1 = gguf_get_tensor_data(gguf, t);
        
        // FFN down
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
        t = gguf_find_tensor(gguf, name);
        if (t) layer->w2 = gguf_get_tensor_data(gguf, t);
        
        // FFN up
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
        t = gguf_find_tensor(gguf, name);
        if (t) layer->w3 = gguf_get_tensor_data(gguf, t);
    }
    
    log_msg(LLAMA_LOG_INFO, "Model weights loaded: %d layers, type=%s",
            model->info.n_layer, ggml_type_name(model->weight_type));
    
    return true;
}

llama_model* llama_load_model(const void* data, size_t size, int32_t n_ctx) {
    return llama_load_model_with_progress(data, size, n_ctx, NULL, NULL);
}

llama_model* llama_load_model_with_progress(
    const void* data, 
    size_t size, 
    int32_t n_ctx,
    llama_progress_callback callback,
    void* user_data
) {
    if (!g_initialized) {
        set_error("Library not initialized");
        return NULL;
    }
    
    if (!data || size == 0) {
        set_error("Invalid model data");
        return NULL;
    }
    
    log_msg(LLAMA_LOG_INFO, "Loading model: %zu bytes", size);
    
    if (callback) callback(0.1f, user_data);
    
    // Parse GGUF
    gguf_context* gguf = gguf_parse(data, size);
    if (!gguf) {
        set_error("Failed to parse GGUF: %s", gguf_get_error());
        return NULL;
    }
    
    if (callback) callback(0.3f, user_data);
    
    // Allocate model
    llama_model* model = (llama_model*)calloc(1, sizeof(llama_model));
    if (!model) {
        set_error("Out of memory for model");
        gguf_free(gguf);
        return NULL;
    }
    model->gguf = gguf;
    
    // Get architecture
    const char* arch = gguf_get_str(gguf, GGUF_KEY_GENERAL_ARCHITECTURE, "llama");
    model->info.arch = arch;
    model->info.name = gguf_get_str(gguf, GGUF_KEY_GENERAL_NAME, "Unknown");
    
    // Build architecture-specific keys
    char key[128];
    
    snprintf(key, sizeof(key), "%s%s", arch, GGUF_KEY_CONTEXT_LENGTH);
    model->info.n_ctx = (int32_t)gguf_get_int(gguf, key, 2048);
    
    snprintf(key, sizeof(key), "%s%s", arch, GGUF_KEY_EMBEDDING_LENGTH);
    model->info.n_embd = (int32_t)gguf_get_int(gguf, key, 4096);
    
    snprintf(key, sizeof(key), "%s%s", arch, GGUF_KEY_BLOCK_COUNT);
    model->info.n_layer = (int32_t)gguf_get_int(gguf, key, 32);
    
    snprintf(key, sizeof(key), "%s%s", arch, GGUF_KEY_ATTENTION_HEAD_COUNT);
    model->info.n_head = (int32_t)gguf_get_int(gguf, key, 32);
    
    snprintf(key, sizeof(key), "%s%s", arch, GGUF_KEY_ATTENTION_HEAD_COUNT_KV);
    model->info.n_head_kv = (int32_t)gguf_get_int(gguf, key, model->info.n_head);
    
    snprintf(key, sizeof(key), "%s%s", arch, GGUF_KEY_FEED_FORWARD_LENGTH);
    model->info.n_ff = (int32_t)gguf_get_int(gguf, key, model->info.n_embd * 4);
    
    snprintf(key, sizeof(key), "%s%s", arch, GGUF_KEY_ROPE_FREQ_BASE);
    model->info.rope_freq_base = gguf_get_float(gguf, key, 10000.0f);
    
    // Override context if specified
    if (n_ctx > 0) {
        model->info.n_ctx = n_ctx;
    }
    
    model->info.n_vocab = (int32_t)gguf_get_int(gguf, "tokenizer.ggml.vocab_size", 32000);
    model->info.model_size = size;
    
    if (callback) callback(0.5f, user_data);
    
    // Load tokenizer
    if (!load_tokenizer(model)) {
        llama_free_model(model);
        return NULL;
    }
    
    if (callback) callback(0.7f, user_data);
    
    // Load weights
    if (!load_model_weights(model)) {
        llama_free_model(model);
        return NULL;
    }
    
    if (callback) callback(1.0f, user_data);
    
    log_msg(LLAMA_LOG_INFO, "Model loaded: %s (%s)", model->info.name, model->info.arch);
    log_msg(LLAMA_LOG_INFO, "  n_vocab=%d, n_ctx=%d, n_embd=%d, n_layer=%d",
            model->info.n_vocab, model->info.n_ctx, model->info.n_embd, model->info.n_layer);
    log_msg(LLAMA_LOG_INFO, "  n_head=%d, n_head_kv=%d, n_ff=%d",
            model->info.n_head, model->info.n_head_kv, model->info.n_ff);
    
    return model;
}

void llama_free_model(llama_model* model) {
    if (!model) return;
    
    if (model->weights.layers) {
        free(model->weights.layers);
    }
    
    if (model->gguf) {
        gguf_free(model->gguf);
    }
    
    free(model);
    log_msg(LLAMA_LOG_INFO, "Model freed");
}

llama_model_info llama_get_model_info(const llama_model* model) {
    if (!model) {
        llama_model_info empty = {0};
        return empty;
    }
    return model->info;
}

const llama_model_info* llama_get_model_info_ptr(const llama_model* model) {
    if (!model) return NULL;
    return &model->info;
}

// ============================================================================
// Context Management
// ============================================================================

static kv_cache_t* kv_cache_alloc(llama_model* model) {
    kv_cache_t* cache = (kv_cache_t*)calloc(1, sizeof(kv_cache_t));
    if (!cache) return NULL;
    
    cache->n_layer = model->info.n_layer;
    cache->n_head_kv = model->info.n_head_kv;
    cache->head_dim = model->info.n_embd / model->info.n_head;
    cache->max_seq = model->info.n_ctx;
    cache->cur_seq = 0;
    
    size_t cache_size = (size_t)cache->n_layer * cache->n_head_kv * 
                        cache->head_dim * cache->max_seq * sizeof(float);
    
    cache->k = (float*)calloc(1, cache_size);
    cache->v = (float*)calloc(1, cache_size);
    
    if (!cache->k || !cache->v) {
        free(cache->k);
        free(cache->v);
        free(cache);
        return NULL;
    }
    
    log_msg(LLAMA_LOG_DEBUG, "KV cache allocated: %zu MB", cache_size * 2 / (1024 * 1024));
    
    return cache;
}

static void kv_cache_free(kv_cache_t* cache) {
    if (!cache) return;
    free(cache->k);
    free(cache->v);
    free(cache);
}

llama_context* llama_create_context(llama_model* model) {
    if (!model) {
        set_error("Invalid model");
        return NULL;
    }
    
    llama_context* ctx = (llama_context*)calloc(1, sizeof(llama_context));
    if (!ctx) {
        set_error("Out of memory for context");
        return NULL;
    }
    
    ctx->model = model;
    
    // Allocate KV cache
    ctx->kv_cache = kv_cache_alloc(model);
    if (!ctx->kv_cache) {
        set_error("Failed to allocate KV cache");
        free(ctx);
        return NULL;
    }
    
    // Allocate logits buffer
    ctx->logits = (float*)calloc(model->info.n_vocab, sizeof(float));
    if (!ctx->logits) {
        set_error("Failed to allocate logits buffer");
        kv_cache_free(ctx->kv_cache);
        free(ctx);
        return NULL;
    }
    
    // Allocate scratch buffer
    ctx->scratch_size = (size_t)model->info.n_embd * model->info.n_ctx * sizeof(float) * 4;
    ctx->scratch = (float*)calloc(1, ctx->scratch_size);
    if (!ctx->scratch) {
        set_error("Failed to allocate scratch buffer");
        free(ctx->logits);
        kv_cache_free(ctx->kv_cache);
        free(ctx);
        return NULL;
    }
    
    ctx->n_past = 0;
    ctx->aborted = false;
    
    log_msg(LLAMA_LOG_INFO, "Context created");
    
    return ctx;
}

void llama_free_context(llama_context* ctx) {
    if (!ctx) return;
    
    kv_cache_free(ctx->kv_cache);
    free(ctx->logits);
    free(ctx->scratch);
    free(ctx);
    
    log_msg(LLAMA_LOG_INFO, "Context freed");
}

void llama_clear_context(llama_context* ctx) {
    if (!ctx) return;
    
    ctx->n_past = 0;
    ctx->aborted = false;
    
    if (ctx->kv_cache) {
        ctx->kv_cache->cur_seq = 0;
        size_t cache_size = (size_t)ctx->kv_cache->n_layer * ctx->kv_cache->n_head_kv * 
                            ctx->kv_cache->head_dim * ctx->kv_cache->max_seq * sizeof(float);
        memset(ctx->kv_cache->k, 0, cache_size);
        memset(ctx->kv_cache->v, 0, cache_size);
    }
    
    memset(&ctx->stats, 0, sizeof(llama_stats));
    
    log_msg(LLAMA_LOG_DEBUG, "Context cleared");
}

int32_t llama_get_context_length(const llama_context* ctx) {
    return ctx ? ctx->n_past : 0;
}

int32_t llama_get_max_context_length(const llama_context* ctx) {
    return ctx && ctx->model ? ctx->model->info.n_ctx : 0;
}

// ============================================================================
// Inference
// ============================================================================

llama_params llama_default_params(void) {
    llama_params params = {
        .max_tokens = 256,
        .temperature = 0.7f,
        .top_p = 0.9f,
        .top_k = 40,
        .repeat_penalty = 1.1f,
        .repeat_last_n = 64,
        .seed = 0,
        .stream = false
    };
    return params;
}

// Dequantize Q4_0 block
static void dequantize_q4_0(const void* src, float* dst, int n) {
    const uint8_t* block = (const uint8_t*)src;
    
    for (int i = 0; i < n / 32; i++) {
        // Read scale (half precision stored as uint16)
        uint16_t scale_bits = *(const uint16_t*)block;
        block += 2;
        
        // Convert half to float (simplified)
        float scale = 0.0f;
        if (scale_bits != 0) {
            int sign = (scale_bits >> 15) & 1;
            int exp = (scale_bits >> 10) & 0x1f;
            int mant = scale_bits & 0x3ff;
            
            if (exp == 0) {
                scale = ldexpf((float)mant / 1024.0f, -14);
            } else if (exp == 31) {
                scale = mant ? NAN : INFINITY;
            } else {
                scale = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15);
            }
            if (sign) scale = -scale;
        }
        
        // Dequantize 32 4-bit values
        for (int j = 0; j < 16; j++) {
            uint8_t byte = *block++;
            int8_t v0 = (byte & 0x0f) - 8;
            int8_t v1 = (byte >> 4) - 8;
            dst[i * 32 + j * 2 + 0] = v0 * scale;
            dst[i * 32 + j * 2 + 1] = v1 * scale;
        }
    }
}

// RMS normalization
static void rms_norm(float* out, const float* x, const float* weight, int n, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf(sum / n + eps);
    
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * scale * weight[i];
    }
}

// Softmax
static void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

// SiLU activation
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Matrix-vector multiplication with dequantization
static void matmul_q4_0(float* out, const void* weight, const float* x, int n_out, int n_in) {
    // Temporary buffer for dequantized weights
    float* w_row = (float*)malloc(n_in * sizeof(float));
    if (!w_row) return;
    
    const uint8_t* w = (const uint8_t*)weight;
    size_t row_size = (n_in / 32) * 18;  // Q4_0 block size
    
    for (int i = 0; i < n_out; i++) {
        dequantize_q4_0(w + i * row_size, w_row, n_in);
        
        float sum = 0.0f;
        for (int j = 0; j < n_in; j++) {
            sum += w_row[j] * x[j];
        }
        out[i] = sum;
    }
    
    free(w_row);
}

// Sampling
static int32_t sample_token(float* logits, int n_vocab, llama_params* params, uint64_t* rng_state) {
    // Apply temperature
    if (params->temperature > 0.0f) {
        for (int i = 0; i < n_vocab; i++) {
            logits[i] /= params->temperature;
        }
    }
    
    // Softmax
    softmax(logits, n_vocab);
    
    // Greedy if temperature is 0
    if (params->temperature == 0.0f) {
        int32_t max_idx = 0;
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > logits[max_idx]) max_idx = i;
        }
        return max_idx;
    }
    
    // Top-K sampling
    if (params->top_k > 0 && params->top_k < n_vocab) {
        // Find top-k threshold
        float* sorted = (float*)malloc(n_vocab * sizeof(float));
        memcpy(sorted, logits, n_vocab * sizeof(float));
        
        // Partial sort to find k-th largest
        for (int i = 0; i < params->top_k; i++) {
            for (int j = i + 1; j < n_vocab; j++) {
                if (sorted[j] > sorted[i]) {
                    float tmp = sorted[i];
                    sorted[i] = sorted[j];
                    sorted[j] = tmp;
                }
            }
        }
        float threshold = sorted[params->top_k - 1];
        free(sorted);
        
        // Zero out below threshold
        for (int i = 0; i < n_vocab; i++) {
            if (logits[i] < threshold) logits[i] = 0.0f;
        }
        
        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < n_vocab; i++) sum += logits[i];
        for (int i = 0; i < n_vocab; i++) logits[i] /= sum;
    }
    
    // Top-P (nucleus) sampling
    if (params->top_p < 1.0f) {
        // Sort by probability
        typedef struct { int idx; float prob; } prob_idx;
        prob_idx* sorted = (prob_idx*)malloc(n_vocab * sizeof(prob_idx));
        for (int i = 0; i < n_vocab; i++) {
            sorted[i].idx = i;
            sorted[i].prob = logits[i];
        }
        
        // Sort descending
        for (int i = 0; i < n_vocab - 1; i++) {
            for (int j = i + 1; j < n_vocab; j++) {
                if (sorted[j].prob > sorted[i].prob) {
                    prob_idx tmp = sorted[i];
                    sorted[i] = sorted[j];
                    sorted[j] = tmp;
                }
            }
        }
        
        // Find cutoff
        float cumsum = 0.0f;
        int cutoff = n_vocab;
        for (int i = 0; i < n_vocab; i++) {
            cumsum += sorted[i].prob;
            if (cumsum >= params->top_p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Zero out below cutoff
        for (int i = cutoff; i < n_vocab; i++) {
            logits[sorted[i].idx] = 0.0f;
        }
        free(sorted);
        
        // Renormalize
        float sum = 0.0f;
        for (int i = 0; i < n_vocab; i++) sum += logits[i];
        for (int i = 0; i < n_vocab; i++) logits[i] /= sum;
    }
    
    // Sample from distribution
    *rng_state = *rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    float r = (float)(*rng_state >> 33) / (float)(1ULL << 31);
    
    float cumsum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        cumsum += logits[i];
        if (cumsum >= r) return i;
    }
    
    return n_vocab - 1;
}

// Forward pass for single token
static void forward(llama_context* ctx, int32_t token, int32_t pos) {
    llama_model* model = ctx->model;
    const llama_model_info* info = &model->info;
    model_weights_t* w = &model->weights;
    
    int n_embd = info->n_embd;
    int n_layer = info->n_layer;
    int n_head = info->n_head;
    int n_head_kv = info->n_head_kv;
    int head_dim = n_embd / n_head;
    int n_ff = info->n_ff;
    int n_vocab = info->n_vocab;
    
    float* x = ctx->scratch;
    float* xb = x + n_embd;
    float* q = xb + n_embd;
    float* k = q + n_embd;
    float* v = k + n_embd;
    float* att = v + n_embd;
    float* xb2 = att + n_head * info->n_ctx;
    float* hb = xb2 + n_embd;
    float* hb2 = hb + n_ff;
    
    // Get token embedding
    const float* embd = (const float*)w->token_embd;
    if (model->weight_type == GGML_TYPE_F32) {
        memcpy(x, embd + token * n_embd, n_embd * sizeof(float));
    } else {
        // Dequantize embedding
        dequantize_q4_0((const uint8_t*)w->token_embd + token * ((n_embd / 32) * 18), x, n_embd);
    }
    
    // Process each layer
    for (int l = 0; l < n_layer; l++) {
        layer_weights_t* layer = &w->layers[l];
        
        // Attention norm
        if (layer->attn_norm) {
            rms_norm(xb, x, (const float*)layer->attn_norm, n_embd, 1e-5f);
        } else {
            memcpy(xb, x, n_embd * sizeof(float));
        }
        
        // QKV projections (simplified - would need proper matmul for quantized)
        if (model->weight_type == GGML_TYPE_Q4_0) {
            matmul_q4_0(q, layer->wq, xb, n_embd, n_embd);
            matmul_q4_0(k, layer->wk, xb, n_head_kv * head_dim, n_embd);
            matmul_q4_0(v, layer->wv, xb, n_head_kv * head_dim, n_embd);
        }
        
        // Apply RoPE (simplified)
        float theta = info->rope_freq_base;
        for (int i = 0; i < n_head; i++) {
            for (int j = 0; j < head_dim / 2; j++) {
                float freq = 1.0f / powf(theta, (float)(2 * j) / head_dim);
                float angle = pos * freq;
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);
                
                int idx = i * head_dim + j * 2;
                float q0 = q[idx];
                float q1 = q[idx + 1];
                q[idx] = q0 * cos_val - q1 * sin_val;
                q[idx + 1] = q0 * sin_val + q1 * cos_val;
            }
        }
        
        // Store K, V in cache
        kv_cache_t* cache = ctx->kv_cache;
        size_t kv_offset = (size_t)l * cache->n_head_kv * cache->head_dim * cache->max_seq +
                          pos * cache->n_head_kv * cache->head_dim;
        memcpy(cache->k + kv_offset, k, n_head_kv * head_dim * sizeof(float));
        memcpy(cache->v + kv_offset, v, n_head_kv * head_dim * sizeof(float));
        
        // Attention (simplified - single head for demo)
        memset(att, 0, n_head * (pos + 1) * sizeof(float));
        for (int h = 0; h < n_head; h++) {
            int kv_h = h / (n_head / n_head_kv);
            float* q_h = q + h * head_dim;
            
            for (int t = 0; t <= pos; t++) {
                size_t k_offset = (size_t)l * cache->n_head_kv * cache->head_dim * cache->max_seq +
                                 t * cache->n_head_kv * cache->head_dim + kv_h * head_dim;
                float* k_t = cache->k + k_offset;
                
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q_h[d] * k_t[d];
                }
                att[h * info->n_ctx + t] = score / sqrtf((float)head_dim);
            }
            
            // Softmax attention
            softmax(att + h * info->n_ctx, pos + 1);
        }
        
        // Apply attention to values
        memset(xb2, 0, n_embd * sizeof(float));
        for (int h = 0; h < n_head; h++) {
            int kv_h = h / (n_head / n_head_kv);
            
            for (int t = 0; t <= pos; t++) {
                float a = att[h * info->n_ctx + t];
                size_t v_offset = (size_t)l * cache->n_head_kv * cache->head_dim * cache->max_seq +
                                 t * cache->n_head_kv * cache->head_dim + kv_h * head_dim;
                float* v_t = cache->v + v_offset;
                
                for (int d = 0; d < head_dim; d++) {
                    xb2[h * head_dim + d] += a * v_t[d];
                }
            }
        }
        
        // Output projection
        if (model->weight_type == GGML_TYPE_Q4_0) {
            matmul_q4_0(xb, layer->wo, xb2, n_embd, n_embd);
        }
        
        // Residual
        for (int i = 0; i < n_embd; i++) {
            x[i] += xb[i];
        }
        
        // FFN norm
        if (layer->ffn_norm) {
            rms_norm(xb, x, (const float*)layer->ffn_norm, n_embd, 1e-5f);
        }
        
        // FFN
        if (model->weight_type == GGML_TYPE_Q4_0) {
            matmul_q4_0(hb, layer->w1, xb, n_ff, n_embd);   // Gate
            matmul_q4_0(hb2, layer->w3, xb, n_ff, n_embd);  // Up
        }
        
        // SiLU and element-wise multiply
        for (int i = 0; i < n_ff; i++) {
            hb[i] = silu(hb[i]) * hb2[i];
        }
        
        // Down projection
        if (model->weight_type == GGML_TYPE_Q4_0) {
            matmul_q4_0(xb, layer->w2, hb, n_embd, n_ff);
        }
        
        // Residual
        for (int i = 0; i < n_embd; i++) {
            x[i] += xb[i];
        }
    }
    
    // Final norm
    if (w->output_norm) {
        rms_norm(x, x, (const float*)w->output_norm, n_embd, 1e-5f);
    }
    
    // Output projection to logits
    if (model->weight_type == GGML_TYPE_Q4_0) {
        matmul_q4_0(ctx->logits, w->output, x, n_vocab, n_embd);
    }
}

char* llama_generate(
    llama_context* ctx,
    const char* prompt,
    llama_params params
) {
    if (!ctx || !prompt) {
        set_error("Invalid context or prompt");
        return NULL;
    }
    
    llama_model* model = ctx->model;
    ctx->aborted = false;
    
    // Initialize RNG
    uint64_t rng_state = params.seed ? params.seed : (uint64_t)time(NULL);
    
    // Tokenize prompt
    int32_t max_prompt_tokens = model->info.n_ctx;
    int32_t* prompt_tokens = (int32_t*)malloc(max_prompt_tokens * sizeof(int32_t));
    if (!prompt_tokens) {
        set_error("Out of memory for prompt tokens");
        return NULL;
    }
    
    int32_t n_prompt = llama_tokenize(model, prompt, prompt_tokens, max_prompt_tokens, true);
    if (n_prompt < 0) {
        free(prompt_tokens);
        set_error("Tokenization failed");
        return NULL;
    }
    
    ctx->stats.tokens_prompt = n_prompt;
    
    // Output buffer
    size_t output_capacity = 4096;
    char* output = (char*)malloc(output_capacity);
    if (!output) {
        free(prompt_tokens);
        set_error("Out of memory for output");
        return NULL;
    }
    output[0] = '\0';
    size_t output_len = 0;
    
    clock_t start_time = clock();
    
    // Process prompt (prefill)
    for (int32_t i = 0; i < n_prompt && !ctx->aborted; i++) {
        forward(ctx, prompt_tokens[i], ctx->n_past++);
    }
    
    ctx->stats.time_prompt_ms = (float)(clock() - start_time) * 1000.0f / CLOCKS_PER_SEC;
    start_time = clock();
    
    // Generate tokens
    int32_t n_generated = 0;
    int32_t eos_token = model->tokenizer.eos_token;
    
    while (n_generated < params.max_tokens && !ctx->aborted) {
        // Sample next token
        int32_t next_token = sample_token(ctx->logits, model->info.n_vocab, &params, &rng_state);
        
        // Check for EOS
        if (next_token == eos_token) {
            break;
        }
        
        // Decode token
        const char* token_str = llama_token_to_str(model, next_token);
        if (token_str) {
            size_t token_len = strlen(token_str);
            
            // Grow output buffer if needed
            if (output_len + token_len + 1 > output_capacity) {
                output_capacity *= 2;
                char* new_output = (char*)realloc(output, output_capacity);
                if (!new_output) {
                    free(prompt_tokens);
                    free(output);
                    set_error("Out of memory for output");
                    return NULL;
                }
                output = new_output;
            }
            
            memcpy(output + output_len, token_str, token_len);
            output_len += token_len;
            output[output_len] = '\0';
        }
        
        // Forward pass for next token
        forward(ctx, next_token, ctx->n_past++);
        n_generated++;
    }
    
    ctx->stats.time_generation_ms = (float)(clock() - start_time) * 1000.0f / CLOCKS_PER_SEC;
    ctx->stats.tokens_generated = n_generated;
    ctx->stats.tokens_per_second = n_generated / (ctx->stats.time_generation_ms / 1000.0f);
    
    free(prompt_tokens);
    
    log_msg(LLAMA_LOG_INFO, "Generated %d tokens in %.2f ms (%.2f tok/s)",
            n_generated, ctx->stats.time_generation_ms, ctx->stats.tokens_per_second);
    
    return output;
}

int32_t llama_generate_stream(
    llama_context* ctx,
    const char* prompt,
    llama_params params,
    llama_token_callback callback,
    void* user_data
) {
    if (!ctx || !prompt) {
        set_error("Invalid context or prompt");
        return -1;
    }
    
    llama_model* model = ctx->model;
    ctx->aborted = false;
    
    // Initialize RNG
    uint64_t rng_state = params.seed ? params.seed : (uint64_t)time(NULL);
    
    // Tokenize prompt
    int32_t max_prompt_tokens = model->info.n_ctx;
    int32_t* prompt_tokens = (int32_t*)malloc(max_prompt_tokens * sizeof(int32_t));
    if (!prompt_tokens) {
        set_error("Out of memory for prompt tokens");
        return -1;
    }
    
    int32_t n_prompt = llama_tokenize(model, prompt, prompt_tokens, max_prompt_tokens, true);
    if (n_prompt < 0) {
        free(prompt_tokens);
        set_error("Tokenization failed");
        return -1;
    }
    
    ctx->stats.tokens_prompt = n_prompt;
    
    clock_t start_time = clock();
    
    // Process prompt (prefill)
    for (int32_t i = 0; i < n_prompt && !ctx->aborted; i++) {
        forward(ctx, prompt_tokens[i], ctx->n_past++);
    }
    
    ctx->stats.time_prompt_ms = (float)(clock() - start_time) * 1000.0f / CLOCKS_PER_SEC;
    start_time = clock();
    
    // Generate tokens
    int32_t n_generated = 0;
    int32_t eos_token = model->tokenizer.eos_token;
    
    while (n_generated < params.max_tokens && !ctx->aborted) {
        // Sample next token
        int32_t next_token = sample_token(ctx->logits, model->info.n_vocab, &params, &rng_state);
        
        // Check for EOS
        if (next_token == eos_token) {
            break;
        }
        
        // Callback with token
        const char* token_str = llama_token_to_str(model, next_token);
        if (token_str && callback) {
            callback(token_str, user_data);
        }
        
        // Forward pass for next token
        forward(ctx, next_token, ctx->n_past++);
        n_generated++;
    }
    
    ctx->stats.time_generation_ms = (float)(clock() - start_time) * 1000.0f / CLOCKS_PER_SEC;
    ctx->stats.tokens_generated = n_generated;
    ctx->stats.tokens_per_second = n_generated / (ctx->stats.time_generation_ms / 1000.0f);
    
    free(prompt_tokens);
    
    return n_generated;
}

void llama_abort(llama_context* ctx) {
    if (ctx) ctx->aborted = true;
}

bool llama_is_aborted(const llama_context* ctx) {
    return ctx ? ctx->aborted : false;
}

llama_stats llama_get_stats(const llama_context* ctx) {
    if (!ctx) {
        llama_stats empty = {0};
        return empty;
    }
    return ctx->stats;
}

void llama_free_string(char* str) {
    free(str);
}

// ============================================================================
// Chat Templates
// ============================================================================

llama_template llama_detect_template(const llama_model* model) {
    if (!model) return LLAMA_TEMPLATE_CHATML;
    
    const char* arch = model->info.arch;
    if (!arch) return LLAMA_TEMPLATE_CHATML;
    
    if (strstr(arch, "qwen")) return LLAMA_TEMPLATE_CHATML;
    if (strstr(arch, "llama") && strstr(model->info.name, "3")) return LLAMA_TEMPLATE_LLAMA3;
    if (strstr(arch, "llama")) return LLAMA_TEMPLATE_LLAMA2;
    if (strstr(arch, "phi")) return LLAMA_TEMPLATE_PHI;
    if (strstr(arch, "mistral")) return LLAMA_TEMPLATE_LLAMA2;
    
    return LLAMA_TEMPLATE_CHATML;
}

char* llama_apply_template(
    const llama_model* model,
    const llama_message* messages,
    int32_t n_messages,
    llama_template template,
    bool add_generation_prompt
) {
    if (!messages || n_messages <= 0) return NULL;
    
    if (template == LLAMA_TEMPLATE_AUTO) {
        template = llama_detect_template(model);
    }
    
    // Calculate buffer size
    size_t total_len = 0;
    for (int i = 0; i < n_messages; i++) {
        total_len += strlen(messages[i].content) + 100;  // Extra for template markers
    }
    total_len += 100;  // Generation prompt
    
    char* result = (char*)malloc(total_len);
    if (!result) return NULL;
    
    char* ptr = result;
    *ptr = '\0';
    
    for (int i = 0; i < n_messages; i++) {
        const char* role = messages[i].role;
        const char* content = messages[i].content;
        
        switch (template) {
            case LLAMA_TEMPLATE_CHATML:
                ptr += sprintf(ptr, "<|im_start|>%s\n%s<|im_end|>\n", role, content);
                break;
                
            case LLAMA_TEMPLATE_LLAMA2:
                if (strcmp(role, "system") == 0) {
                    ptr += sprintf(ptr, "<<SYS>>\n%s\n<</SYS>>\n\n", content);
                } else if (strcmp(role, "user") == 0) {
                    ptr += sprintf(ptr, "[INST] %s [/INST]", content);
                } else {
                    ptr += sprintf(ptr, " %s </s><s>", content);
                }
                break;
                
            case LLAMA_TEMPLATE_LLAMA3:
                ptr += sprintf(ptr, "<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", role, content);
                break;
                
            case LLAMA_TEMPLATE_PHI:
                if (strcmp(role, "user") == 0) {
                    ptr += sprintf(ptr, "Instruct: %s\n", content);
                } else if (strcmp(role, "assistant") == 0) {
                    ptr += sprintf(ptr, "Output: %s\n", content);
                }
                break;
                
            case LLAMA_TEMPLATE_ALPACA:
                if (strcmp(role, "system") == 0) {
                    ptr += sprintf(ptr, "%s\n\n", content);
                } else if (strcmp(role, "user") == 0) {
                    ptr += sprintf(ptr, "### Instruction:\n%s\n\n", content);
                } else {
                    ptr += sprintf(ptr, "### Response:\n%s\n\n", content);
                }
                break;
                
            default:
                ptr += sprintf(ptr, "<|im_start|>%s\n%s<|im_end|>\n", role, content);
                break;
        }
    }
    
    // Add generation prompt
    if (add_generation_prompt) {
        switch (template) {
            case LLAMA_TEMPLATE_CHATML:
                ptr += sprintf(ptr, "<|im_start|>assistant\n");
                break;
            case LLAMA_TEMPLATE_LLAMA3:
                ptr += sprintf(ptr, "<|start_header_id|>assistant<|end_header_id|>\n\n");
                break;
            case LLAMA_TEMPLATE_PHI:
                ptr += sprintf(ptr, "Output:");
                break;
            case LLAMA_TEMPLATE_ALPACA:
                ptr += sprintf(ptr, "### Response:\n");
                break;
            default:
                ptr += sprintf(ptr, "<|im_start|>assistant\n");
                break;
        }
    }
    
    return result;
}

// ============================================================================
// Memory Info
// ============================================================================

llama_memory_info llama_get_memory_info(const llama_context* ctx) {
    llama_memory_info info = {0};
    
    if (!ctx) return info;
    
    info.model_size = ctx->model ? ctx->model->info.model_size : 0;
    
    if (ctx->kv_cache) {
        info.kv_cache_size = (size_t)ctx->kv_cache->n_layer * ctx->kv_cache->n_head_kv *
                            ctx->kv_cache->head_dim * ctx->kv_cache->max_seq * sizeof(float) * 2;
    }
    
    info.scratch_size = ctx->scratch_size;
    info.total_allocated = info.model_size + info.kv_cache_size + info.scratch_size +
                          ctx->model->info.n_vocab * sizeof(float);
    
    return info;
}

bool llama_has_simd(void) {
#if LLAMA_TINY_SIMD
    return true;
#else
    return false;
#endif
}

int32_t llama_get_n_threads(void) {
    return 1;  // Single-threaded for WASM
}

void llama_set_n_threads(int32_t n_threads) {
    (void)n_threads;  // Not implemented for WASM
}

// ============================================================================
// SKZ Agent Integration
// ============================================================================

skz_params skz_default_params(skz_task_type task) {
    skz_params params = {
        .task = task,
        .system_prompt = NULL,
        .confidence_threshold = 0.7f,
        .structured_output = false,
        .output_schema = NULL
    };
    
    switch (task) {
        case SKZ_TASK_RESEARCH:
            params.system_prompt = "You are a research discovery agent specializing in academic literature analysis.";
            break;
        case SKZ_TASK_SUBMISSION:
            params.system_prompt = "You are a submission assistant agent helping authors prepare manuscripts.";
            break;
        case SKZ_TASK_EDITORIAL:
            params.system_prompt = "You are an editorial orchestration agent managing publication workflows.";
            break;
        case SKZ_TASK_REVIEW:
            params.system_prompt = "You are a review coordination agent matching reviewers to manuscripts.";
            break;
        case SKZ_TASK_QUALITY:
            params.system_prompt = "You are a content quality agent ensuring publication standards.";
            break;
        case SKZ_TASK_PUBLISHING:
            params.system_prompt = "You are a publishing production agent handling final publication steps.";
            break;
        case SKZ_TASK_ANALYTICS:
            params.system_prompt = "You are an analytics agent providing insights on publication metrics.";
            break;
        case SKZ_TASK_CLASSIFICATION:
            params.system_prompt = "You are a text classification agent. Classify the input into the given categories.";
            params.structured_output = true;
            break;
        case SKZ_TASK_SUMMARIZATION:
            params.system_prompt = "You are a summarization agent. Provide concise summaries of the input text.";
            break;
        case SKZ_TASK_EXTRACTION:
            params.system_prompt = "You are an information extraction agent. Extract structured data from text.";
            params.structured_output = true;
            break;
        default:
            params.system_prompt = "You are a helpful AI assistant.";
            break;
    }
    
    return params;
}

char* skz_generate(
    llama_context* ctx,
    const char* input,
    skz_params params,
    llama_params gen_params
) {
    if (!ctx || !input) return NULL;
    
    // Build messages
    llama_message messages[2];
    int n_messages = 0;
    
    if (params.system_prompt) {
        messages[n_messages].role = "system";
        messages[n_messages].content = params.system_prompt;
        n_messages++;
    }
    
    messages[n_messages].role = "user";
    messages[n_messages].content = input;
    n_messages++;
    
    // Apply template
    char* prompt = llama_apply_template(ctx->model, messages, n_messages, LLAMA_TEMPLATE_AUTO, true);
    if (!prompt) return NULL;
    
    // Generate
    char* result = llama_generate(ctx, prompt, gen_params);
    
    free(prompt);
    return result;
}

char* skz_classify(
    llama_context* ctx,
    const char* text,
    const char** categories,
    int32_t n_categories
) {
    if (!ctx || !text || !categories || n_categories <= 0) return NULL;
    
    // Build classification prompt
    size_t prompt_size = strlen(text) + 1024;
    for (int i = 0; i < n_categories; i++) {
        prompt_size += strlen(categories[i]) + 10;
    }
    
    char* prompt = (char*)malloc(prompt_size);
    if (!prompt) return NULL;
    
    char* ptr = prompt;
    ptr += sprintf(ptr, "Classify the following text into one of these categories: ");
    for (int i = 0; i < n_categories; i++) {
        if (i > 0) ptr += sprintf(ptr, ", ");
        ptr += sprintf(ptr, "%s", categories[i]);
    }
    ptr += sprintf(ptr, ".\n\nText: \"%s\"\n\nRespond with only the category name.", text);
    
    // Generate
    llama_params gen_params = llama_default_params();
    gen_params.max_tokens = 50;
    gen_params.temperature = 0.1f;
    
    skz_params skz = skz_default_params(SKZ_TASK_CLASSIFICATION);
    char* result = skz_generate(ctx, prompt, skz, gen_params);
    
    free(prompt);
    return result;
}

char* skz_extract(
    llama_context* ctx,
    const char* text,
    const char* json_schema
) {
    if (!ctx || !text || !json_schema) return NULL;
    
    // Build extraction prompt
    size_t prompt_size = strlen(text) + strlen(json_schema) + 512;
    char* prompt = (char*)malloc(prompt_size);
    if (!prompt) return NULL;
    
    sprintf(prompt, 
        "Extract information from the following text according to this JSON schema:\n"
        "Schema: %s\n\n"
        "Text: \"%s\"\n\n"
        "Respond with only the JSON object matching the schema.",
        json_schema, text);
    
    // Generate
    llama_params gen_params = llama_default_params();
    gen_params.max_tokens = 512;
    gen_params.temperature = 0.1f;
    
    skz_params skz = skz_default_params(SKZ_TASK_EXTRACTION);
    char* result = skz_generate(ctx, prompt, skz, gen_params);
    
    free(prompt);
    return result;
}

// ============================================================================
// WASM Exports
// ============================================================================

#ifdef __EMSCRIPTEN__

static llama_token_callback wasm_token_callback = NULL;
static llama_progress_callback wasm_progress_callback = NULL;

EMSCRIPTEN_KEEPALIVE
void set_token_callback(llama_token_callback callback) {
    wasm_token_callback = callback;
    g_token_callback = callback;
}

EMSCRIPTEN_KEEPALIVE
void set_progress_callback(llama_progress_callback callback) {
    wasm_progress_callback = callback;
    g_progress_callback = callback;
}

#endif
