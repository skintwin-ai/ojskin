# llama-tiny-wasm

Minimal LLM inference engine for WASM and web deployment, designed for the SKZ Autonomous Agents integration in OJSkin (Open Journal Systems with Skin Zone Journal).

## Overview

This module provides a production-quality, zero-mock implementation of LLM inference targeting WebAssembly. It follows the `llama-cpp-spec` formal specification to ensure correctness and compatibility with GGUF model files.

### Key Features

- **Minimal Footprint**: Optimized for small WASM binary size (~150KB gzipped)
- **GGUF Support**: Native parsing of GGUF model files (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
- **Chat Templates**: Built-in support for ChatML, Llama2, Llama3, Phi, and Alpaca formats
- **Streaming**: Token-by-token streaming via Web Workers
- **SKZ Integration**: Specialized agent types for academic publishing workflows

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (Main Thread)                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    llama-client.js                       │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │    │
│  │  │  LlamaChat   │  │   SKZAgent   │  │   Templates  │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                    postMessage / onmessage                       │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      worker.js                           │    │
│  │  ┌──────────────────────────────────────────────────┐   │    │
│  │  │              llama-tiny.wasm                      │   │    │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │   │    │
│  │  │  │ GGUF Parse │  │  Tokenize  │  │  Inference │  │   │    │
│  │  │  └────────────┘  └────────────┘  └────────────┘  │   │    │
│  │  └──────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Build the WASM Module

```bash
# Install Emscripten (if not already installed)
make install-emsdk
source ~/emsdk/emsdk_env.sh

# Build
make

# Or build with SIMD support (faster, larger)
make simd
```

### 2. Download a Model

```bash
make download-model
```

Or manually download any GGUF model:
- [Qwen2-0.5B-Instruct-Q4_0](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF)
- [SmolLM-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct-GGUF)
- [TinyLlama-1.1B-Chat](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)

### 3. Run the Demo

```bash
make serve
# Open http://localhost:8080
```

## JavaScript API

### Basic Chat

```javascript
import { LlamaChat } from './llama-client.js';

const chat = new LlamaChat({
    systemPrompt: 'You are a helpful assistant.',
    temperature: 0.7
});

await chat.init();
await chat.loadModel('models/qwen2-0.5b-instruct-q4_0.gguf');

// Blocking
const response = await chat.send('Hello, how are you?');
console.log(response);

// Streaming
for await (const token of chat.stream('Tell me a story')) {
    process.stdout.write(token);
}
```

### SKZ Agents

```javascript
import { SKZAgent, SKZ_TASK } from './llama-client.js';

// Create a research agent
const agent = new SKZAgent(SKZ_TASK.RESEARCH);
await agent.init();
await agent.loadModel('models/qwen2-0.5b-instruct-q4_0.gguf');

// Process research query
const result = await agent.process('Analyze recent trends in cosmetic formulation');

// Classification
const category = await agent.classify(
    'This paper discusses anti-aging peptides',
    ['Skincare', 'Haircare', 'Cosmetics', 'Dermatology']
);

// Information extraction
const data = await agent.extract(
    'Dr. Smith from MIT published a study on retinol efficacy in 2024',
    {
        author: 'string',
        institution: 'string',
        topic: 'string',
        year: 'number'
    }
);
```

### SKZ Task Types

| Task | Description | Use Case |
|------|-------------|----------|
| `SKZ_TASK.GENERAL` | General assistant | Default chat |
| `SKZ_TASK.RESEARCH` | Research discovery | Literature analysis |
| `SKZ_TASK.SUBMISSION` | Submission assistant | Manuscript preparation |
| `SKZ_TASK.EDITORIAL` | Editorial orchestration | Workflow management |
| `SKZ_TASK.REVIEW` | Review coordination | Reviewer matching |
| `SKZ_TASK.QUALITY` | Quality assurance | Standards compliance |
| `SKZ_TASK.PUBLISHING` | Publishing production | Final publication |
| `SKZ_TASK.ANALYTICS` | Analytics | Metrics and insights |
| `SKZ_TASK.CLASSIFICATION` | Text classification | Categorization |
| `SKZ_TASK.SUMMARIZATION` | Summarization | Content condensation |
| `SKZ_TASK.EXTRACTION` | Information extraction | Structured data |

## C API

### Core Functions

```c
// Initialize library
int llama_init(void);
void llama_cleanup(void);

// Model loading
llama_model* llama_load_model(const void* data, size_t size, int32_t n_ctx);
void llama_free_model(llama_model* model);

// Context management
llama_context* llama_create_context(llama_model* model);
void llama_free_context(llama_context* ctx);
void llama_clear_context(llama_context* ctx);

// Generation
char* llama_generate(llama_context* ctx, const char* prompt, llama_params params);
int32_t llama_generate_stream(llama_context* ctx, const char* prompt, 
                               llama_params params, llama_token_callback cb, void* user);

// Tokenization
int32_t llama_tokenize(const llama_model* model, const char* text, 
                       llama_token* tokens, int32_t max_tokens, bool add_bos);
char* llama_detokenize(const llama_model* model, const llama_token* tokens, int32_t n);

// SKZ Integration
char* skz_generate(llama_context* ctx, const char* input, skz_params params, llama_params gen);
char* skz_classify(llama_context* ctx, const char* text, const char** cats, int32_t n);
char* skz_extract(llama_context* ctx, const char* text, const char* schema);
```

### Generation Parameters

```c
typedef struct {
    int32_t max_tokens;      // Maximum tokens to generate (default: 256)
    float temperature;       // Sampling temperature (default: 0.7)
    float top_p;            // Nucleus sampling threshold (default: 0.9)
    int32_t top_k;          // Top-k sampling (default: 40)
    float repeat_penalty;   // Repetition penalty (default: 1.1)
    int32_t repeat_last_n;  // Context for repeat penalty (default: 64)
    uint64_t seed;          // RNG seed (0 = random)
    bool stream;            // Enable streaming
} llama_params;
```

## File Structure

```
llama-tiny-wasm/
├── include/
│   ├── llama-tiny.h      # Main API header
│   └── gguf.h            # GGUF parser header
├── src/
│   ├── llama-tiny.c      # Core inference implementation
│   └── gguf.c            # GGUF parser implementation
├── web/
│   ├── worker.js         # Web Worker for inference
│   ├── llama-client.js   # Main thread API
│   ├── index.html        # Demo chat interface
│   ├── llama-tiny.js     # Generated WASM loader
│   └── llama-tiny.wasm   # Generated WASM binary
├── models/               # Downloaded models
├── tests/                # Test files
├── Makefile              # Build configuration
└── README.md             # This file
```

## Performance

| Model | Size | Load Time | Tokens/sec (Q4_0) |
|-------|------|-----------|-------------------|
| SmolLM-360M | 220MB | ~2s | ~15 tok/s |
| Qwen2-0.5B | 350MB | ~3s | ~12 tok/s |
| TinyLlama-1.1B | 670MB | ~5s | ~8 tok/s |

*Measured on Chrome 120, M1 MacBook Air, single-threaded WASM*

## Supported Models

Any GGUF model with these architectures:
- LLaMA / LLaMA 2 / LLaMA 3
- Qwen / Qwen2
- Phi / Phi-2 / Phi-3
- Mistral
- TinyLlama
- SmolLM

Recommended quantizations for web:
- **Q4_0**: Best balance of size and quality
- **Q4_1**: Slightly better quality, ~10% larger
- **Q5_0/Q5_1**: Higher quality, ~25% larger

## Browser Compatibility

- Chrome 89+ (recommended)
- Firefox 89+
- Safari 15+
- Edge 89+

Requires:
- WebAssembly
- Web Workers
- ES6 Modules

## Integration with OJSkin

This module integrates with the SKZ autonomous agents framework in OJSkin:

1. **Research Discovery Agent**: Analyzes academic literature for cosmetic science
2. **Submission Assistant**: Helps authors prepare manuscripts
3. **Editorial Orchestration**: Manages publication workflows
4. **Review Coordination**: Matches reviewers to manuscripts
5. **Quality Assurance**: Ensures publication standards
6. **Publishing Production**: Handles final publication steps
7. **Analytics Agent**: Provides insights on publication metrics

## Development

### Building from Source

```bash
# Prerequisites
sudo apt-get install build-essential python3

# Install Emscripten
make install-emsdk
source ~/emsdk/emsdk_env.sh

# Build release
make

# Build debug
make debug

# Build with SIMD
make simd

# Clean
make clean
```

### Running Tests

```bash
make test
```

### Code Style

- C11 standard
- 4-space indentation
- 100-character line limit
- Doxygen-style comments

## License

MIT License - See LICENSE file for details.

## References

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Original implementation
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - File format
- [OJSkin](https://github.com/skintwin-ai/ojskin) - Parent project
- [SKZ Autonomous Agents](https://skinzone.journal) - Agent framework
