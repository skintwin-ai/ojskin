# Makefile for llama-tiny-wasm
# SKZ Autonomous Agents Integration
# OJSkin - Open Journal Systems with Skin Zone Journal

# Compiler settings
CC = emcc
CFLAGS = -O3 -Wall -Wextra -std=c11
LDFLAGS = 

# WASM settings
WASM_FLAGS = -s WASM=1 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s MAXIMUM_MEMORY=4GB \
	-s STACK_SIZE=1MB \
	-s TOTAL_MEMORY=64MB \
	-s MODULARIZE=1 \
	-s EXPORT_ES6=0 \
	-s EXPORT_NAME='createLlamaTinyModule' \
	-s SINGLE_FILE=0 \
	--no-entry

# Exported functions
EXPORTED_FUNCTIONS = '[ \
	"_malloc", \
	"_free", \
	"_llama_init", \
	"_llama_cleanup", \
	"_llama_load_model", \
	"_llama_load_model_with_progress", \
	"_llama_free_model", \
	"_llama_get_model_info", \
	"_llama_get_model_info_ptr", \
	"_llama_create_context", \
	"_llama_free_context", \
	"_llama_clear_context", \
	"_llama_get_context_length", \
	"_llama_get_max_context_length", \
	"_llama_default_params", \
	"_llama_generate", \
	"_llama_generate_stream", \
	"_llama_abort", \
	"_llama_is_aborted", \
	"_llama_get_stats", \
	"_llama_free_string", \
	"_llama_tokenize", \
	"_llama_detokenize", \
	"_llama_token_to_str", \
	"_llama_token_bos", \
	"_llama_token_eos", \
	"_llama_token_pad", \
	"_llama_apply_template", \
	"_llama_detect_template", \
	"_llama_version", \
	"_llama_get_error", \
	"_llama_clear_error", \
	"_llama_get_memory_info", \
	"_llama_has_simd", \
	"_llama_get_n_threads", \
	"_llama_set_n_threads", \
	"_skz_default_params", \
	"_skz_generate", \
	"_skz_classify", \
	"_skz_extract", \
	"_set_token_callback", \
	"_set_progress_callback" \
]'

EXPORTED_RUNTIME_METHODS = '["ccall", "cwrap", "UTF8ToString", "allocateUTF8", "getValue", "setValue", "addFunction", "removeFunction"]'

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
WEB_DIR = web

# Source files
SRCS = $(SRC_DIR)/llama-tiny.c $(SRC_DIR)/gguf.c
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Output files
WASM_OUT = $(WEB_DIR)/llama-tiny.js
WASM_BIN = $(WEB_DIR)/llama-tiny.wasm

# Include paths
INCLUDES = -I$(INC_DIR)

# Default target
.PHONY: all
all: $(WASM_OUT)

# WASM build (minimal, no SIMD)
$(WASM_OUT): $(SRCS)
	@mkdir -p $(WEB_DIR)
	$(CC) $(CFLAGS) $(WASM_FLAGS) \
		-s EXPORTED_FUNCTIONS=$(EXPORTED_FUNCTIONS) \
		-s EXPORTED_RUNTIME_METHODS=$(EXPORTED_RUNTIME_METHODS) \
		-s ALLOW_TABLE_GROWTH=1 \
		$(INCLUDES) \
		$(SRCS) \
		-o $@
	@echo "Built: $@ ($(shell ls -lh $@ | awk '{print $$5}'))"
	@echo "Built: $(WASM_BIN) ($(shell ls -lh $(WASM_BIN) | awk '{print $$5}'))"

# WASM build with SIMD (faster, larger)
.PHONY: simd
simd: CFLAGS += -msimd128
simd: $(WASM_OUT)
	@echo "Built with SIMD support"

# Debug build
.PHONY: debug
debug: CFLAGS = -O0 -g -Wall -Wextra -std=c11
debug: WASM_FLAGS += -s ASSERTIONS=2 -s SAFE_HEAP=1
debug: $(WASM_OUT)
	@echo "Built debug version"

# Clean
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(WEB_DIR)/llama-tiny.js $(WEB_DIR)/llama-tiny.wasm

# Development server
.PHONY: serve
serve:
	@echo "Starting development server at http://localhost:8080"
	@cd $(WEB_DIR) && python3 -m http.server 8080

# Install Emscripten (if needed)
.PHONY: install-emsdk
install-emsdk:
	@if [ ! -d "$(HOME)/emsdk" ]; then \
		echo "Installing Emscripten SDK..."; \
		cd $(HOME) && git clone https://github.com/emscripten-core/emsdk.git; \
		cd $(HOME)/emsdk && ./emsdk install latest && ./emsdk activate latest; \
	fi
	@echo "Emscripten SDK installed at $(HOME)/emsdk"
	@echo "Run: source $(HOME)/emsdk/emsdk_env.sh"

# Download test model
.PHONY: download-model
download-model:
	@mkdir -p models
	@echo "Downloading Qwen2-0.5B-Instruct-Q4_0..."
	@wget -c -O models/qwen2-0.5b-instruct-q4_0.gguf \
		"https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_0.gguf"
	@echo "Model downloaded to models/qwen2-0.5b-instruct-q4_0.gguf"

# Run tests
.PHONY: test
test: $(WASM_OUT)
	@echo "Running tests..."
	@node --experimental-wasm-modules tests/test_basic.mjs

# Help
.PHONY: help
help:
	@echo "llama-tiny-wasm Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build WASM module (default)"
	@echo "  simd           - Build with SIMD support (faster, larger)"
	@echo "  debug          - Build debug version"
	@echo "  clean          - Remove build artifacts"
	@echo "  serve          - Start development server"
	@echo "  install-emsdk  - Install Emscripten SDK"
	@echo "  download-model - Download test model (Qwen2-0.5B)"
	@echo "  test           - Run tests"
	@echo "  help           - Show this help"
	@echo ""
	@echo "Requirements:"
	@echo "  - Emscripten SDK (run 'make install-emsdk' to install)"
	@echo "  - Source emsdk_env.sh before building"
	@echo ""
	@echo "Example:"
	@echo "  source ~/emsdk/emsdk_env.sh"
	@echo "  make"
	@echo "  make serve"
