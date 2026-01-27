/**
 * llama-tiny.h - Minimal LLM inference for WASM
 * 
 * SKZ Autonomous Agents Integration
 * OJSkin - Open Journal Systems with Skin Zone Journal
 * 
 * Single-header API for tiny LLM inference engines.
 * Designed for browser web workers, edge computing, and SKZ agent integration.
 * 
 * Zero tolerance for mock implementations - production quality only.
 */

#ifndef LLAMA_TINY_H
#define LLAMA_TINY_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Version and Build Info
// ============================================================================

#define LLAMA_TINY_VERSION_MAJOR 1
#define LLAMA_TINY_VERSION_MINOR 0
#define LLAMA_TINY_VERSION_PATCH 0
#define LLAMA_TINY_VERSION "1.0.0"

// Build configuration
#ifdef __EMSCRIPTEN__
    #define LLAMA_TINY_WASM 1
#else
    #define LLAMA_TINY_WASM 0
#endif

#ifdef __wasm_simd128__
    #define LLAMA_TINY_SIMD 1
#else
    #define LLAMA_TINY_SIMD 0
#endif

// ============================================================================
// Types
// ============================================================================

typedef int32_t llama_token;

typedef struct llama_model llama_model;
typedef struct llama_context llama_context;

// Token callback for streaming
typedef void (*llama_token_callback)(const char* text, void* user_data);

// Progress callback for model loading
typedef void (*llama_progress_callback)(float progress, void* user_data);

// Log callback
typedef void (*llama_log_callback)(const char* msg, int level, void* user_data);

// Log levels
typedef enum {
    LLAMA_LOG_DEBUG = 0,
    LLAMA_LOG_INFO = 1,
    LLAMA_LOG_WARN = 2,
    LLAMA_LOG_ERROR = 3
} llama_log_level;

// Generation parameters
typedef struct {
    int32_t  max_tokens;      // Maximum tokens to generate (default: 256)
    float    temperature;     // Sampling temperature (default: 0.7)
    float    top_p;           // Nucleus sampling threshold (default: 0.9)
    int32_t  top_k;           // Top-K sampling (default: 40)
    float    repeat_penalty;  // Repetition penalty (default: 1.1)
    int32_t  repeat_last_n;   // Tokens to consider for repetition (default: 64)
    uint64_t seed;            // RNG seed (0 = random)
    bool     stream;          // Enable streaming output
} llama_params;

// Model info
typedef struct {
    const char* name;         // Model name from GGUF
    const char* arch;         // Architecture (llama, phi, qwen2, etc.)
    int32_t n_vocab;          // Vocabulary size
    int32_t n_ctx;            // Context length
    int32_t n_embd;           // Embedding dimension
    int32_t n_layer;          // Number of layers
    int32_t n_head;           // Number of attention heads
    int32_t n_head_kv;        // Number of KV heads (for GQA)
    int32_t n_ff;             // Feed-forward dimension
    float   rope_freq_base;   // RoPE frequency base
    size_t  model_size;       // Model size in bytes
} llama_model_info;

// Generation statistics
typedef struct {
    int32_t tokens_generated;
    int32_t tokens_prompt;
    float   time_prompt_ms;
    float   time_generation_ms;
    float   tokens_per_second;
} llama_stats;

// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialize the library.
 * Must be called before any other function.
 * Returns 0 on success, -1 on error.
 */
int llama_init(void);

/**
 * Cleanup and free all resources.
 */
void llama_cleanup(void);

/**
 * Set log callback (default: stderr).
 */
void llama_set_log_callback(llama_log_callback callback, void* user_data);

/**
 * Set log level threshold.
 */
void llama_set_log_level(llama_log_level level);

// ============================================================================
// Model Loading
// ============================================================================

/**
 * Load model from memory buffer (GGUF format).
 * 
 * @param data    Pointer to GGUF data
 * @param size    Size of data in bytes
 * @param n_ctx   Context size (0 = use model default)
 * @return        Model handle, or NULL on error
 */
llama_model* llama_load_model(const void* data, size_t size, int32_t n_ctx);

/**
 * Load model with progress callback.
 */
llama_model* llama_load_model_with_progress(
    const void* data, 
    size_t size, 
    int32_t n_ctx,
    llama_progress_callback callback,
    void* user_data
);

/**
 * Free model and associated resources.
 */
void llama_free_model(llama_model* model);

/**
 * Get model information.
 */
llama_model_info llama_get_model_info(const llama_model* model);

/**
 * Get model info pointer (for WASM bindings).
 */
const llama_model_info* llama_get_model_info_ptr(const llama_model* model);

// ============================================================================
// Context Management
// ============================================================================

/**
 * Create inference context for a model.
 * Multiple contexts can share the same model.
 */
llama_context* llama_create_context(llama_model* model);

/**
 * Free context.
 */
void llama_free_context(llama_context* ctx);

/**
 * Clear KV cache (start new conversation).
 */
void llama_clear_context(llama_context* ctx);

/**
 * Get current context length (tokens processed).
 */
int32_t llama_get_context_length(const llama_context* ctx);

/**
 * Get maximum context length.
 */
int32_t llama_get_max_context_length(const llama_context* ctx);

// ============================================================================
// Generation
// ============================================================================

/**
 * Get default generation parameters.
 */
llama_params llama_default_params(void);

/**
 * Generate text from prompt (blocking).
 * 
 * @param ctx     Inference context
 * @param prompt  Input prompt (UTF-8)
 * @param params  Generation parameters
 * @return        Generated text (caller must free with llama_free_string)
 */
char* llama_generate(
    llama_context* ctx,
    const char* prompt,
    llama_params params
);

/**
 * Generate text with streaming callback.
 * 
 * @param ctx       Inference context
 * @param prompt    Input prompt (UTF-8)
 * @param params    Generation parameters
 * @param callback  Called for each generated token
 * @param user_data Passed to callback
 * @return          Total tokens generated
 */
int32_t llama_generate_stream(
    llama_context* ctx,
    const char* prompt,
    llama_params params,
    llama_token_callback callback,
    void* user_data
);

/**
 * Abort ongoing generation.
 */
void llama_abort(llama_context* ctx);

/**
 * Check if generation was aborted.
 */
bool llama_is_aborted(const llama_context* ctx);

/**
 * Get generation statistics.
 */
llama_stats llama_get_stats(const llama_context* ctx);

/**
 * Free string returned by llama_generate.
 */
void llama_free_string(char* str);

// ============================================================================
// Tokenization
// ============================================================================

/**
 * Tokenize text.
 * 
 * @param model   Model (for vocabulary)
 * @param text    Input text (UTF-8)
 * @param tokens  Output token buffer
 * @param max_tokens  Buffer size
 * @param add_bos Add beginning-of-sequence token
 * @return        Number of tokens, or negative on error
 */
int32_t llama_tokenize(
    const llama_model* model,
    const char* text,
    llama_token* tokens,
    int32_t max_tokens,
    bool add_bos
);

/**
 * Detokenize tokens to text.
 * 
 * @param model   Model (for vocabulary)
 * @param tokens  Input tokens
 * @param n_tokens Number of tokens
 * @return        Text (caller must free with llama_free_string)
 */
char* llama_detokenize(
    const llama_model* model,
    const llama_token* tokens,
    int32_t n_tokens
);

/**
 * Get token string representation.
 */
const char* llama_token_to_str(const llama_model* model, llama_token token);

/**
 * Get special token IDs.
 */
llama_token llama_token_bos(const llama_model* model);
llama_token llama_token_eos(const llama_model* model);
llama_token llama_token_pad(const llama_model* model);

// ============================================================================
// Chat Templates
// ============================================================================

typedef enum {
    LLAMA_TEMPLATE_AUTO,      // Auto-detect from model
    LLAMA_TEMPLATE_CHATML,    // <|im_start|>...<|im_end|>
    LLAMA_TEMPLATE_LLAMA2,    // [INST]...[/INST]
    LLAMA_TEMPLATE_LLAMA3,    // <|start_header_id|>...
    LLAMA_TEMPLATE_PHI,       // Instruct:...Output:
    LLAMA_TEMPLATE_ALPACA,    // ### Instruction:...
} llama_template;

typedef struct {
    const char* role;     // "system", "user", "assistant"
    const char* content;  // Message content
} llama_message;

/**
 * Apply chat template to messages.
 * 
 * @param model     Model (for template detection)
 * @param messages  Array of messages
 * @param n_messages Number of messages
 * @param template  Template to use (AUTO = detect from model)
 * @param add_generation_prompt  Add assistant prefix for generation
 * @return          Formatted prompt (caller must free)
 */
char* llama_apply_template(
    const llama_model* model,
    const llama_message* messages,
    int32_t n_messages,
    llama_template template,
    bool add_generation_prompt
);

/**
 * Detect template from model architecture.
 */
llama_template llama_detect_template(const llama_model* model);

// ============================================================================
// SKZ Agent Integration
// ============================================================================

/**
 * SKZ Agent task types for specialized inference.
 */
typedef enum {
    SKZ_TASK_GENERAL,           // General text generation
    SKZ_TASK_RESEARCH,          // Research discovery agent
    SKZ_TASK_SUBMISSION,        // Submission assistant agent
    SKZ_TASK_EDITORIAL,         // Editorial orchestration agent
    SKZ_TASK_REVIEW,            // Review coordination agent
    SKZ_TASK_QUALITY,           // Content quality agent
    SKZ_TASK_PUBLISHING,        // Publishing production agent
    SKZ_TASK_ANALYTICS,         // Analytics & monitoring agent
    SKZ_TASK_CLASSIFICATION,    // Text classification
    SKZ_TASK_SUMMARIZATION,     // Text summarization
    SKZ_TASK_EXTRACTION,        // Information extraction
} skz_task_type;

/**
 * SKZ-optimized generation parameters.
 */
typedef struct {
    skz_task_type task;
    const char* system_prompt;
    float confidence_threshold;
    bool structured_output;
    const char* output_schema;  // JSON schema for structured output
} skz_params;

/**
 * Get default SKZ parameters for a task type.
 */
skz_params skz_default_params(skz_task_type task);

/**
 * Generate with SKZ agent optimization.
 */
char* skz_generate(
    llama_context* ctx,
    const char* input,
    skz_params params,
    llama_params gen_params
);

/**
 * Classify text into categories.
 * Returns JSON array of {category, score} pairs.
 */
char* skz_classify(
    llama_context* ctx,
    const char* text,
    const char** categories,
    int32_t n_categories
);

/**
 * Extract structured information from text.
 * Returns JSON object matching the provided schema.
 */
char* skz_extract(
    llama_context* ctx,
    const char* text,
    const char* json_schema
);

// ============================================================================
// Utilities
// ============================================================================

/**
 * Get library version string.
 */
const char* llama_version(void);

/**
 * Get last error message.
 */
const char* llama_get_error(void);

/**
 * Clear error state.
 */
void llama_clear_error(void);

/**
 * Get memory usage statistics.
 */
typedef struct {
    size_t model_size;
    size_t kv_cache_size;
    size_t scratch_size;
    size_t total_allocated;
} llama_memory_info;

llama_memory_info llama_get_memory_info(const llama_context* ctx);

/**
 * Check if SIMD is available.
 */
bool llama_has_simd(void);

/**
 * Get number of available threads.
 */
int32_t llama_get_n_threads(void);

/**
 * Set number of threads for inference.
 */
void llama_set_n_threads(int32_t n_threads);

// ============================================================================
// WASM-specific exports
// ============================================================================

#ifdef __EMSCRIPTEN__

#include <emscripten.h>

// Callback registration for WASM
EMSCRIPTEN_KEEPALIVE void set_token_callback(llama_token_callback callback);
EMSCRIPTEN_KEEPALIVE void set_progress_callback(llama_progress_callback callback);

#endif

#ifdef __cplusplus
}
#endif

#endif // LLAMA_TINY_H
