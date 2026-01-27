/**
 * llama-client.js - Main thread API for LLM chat
 * 
 * SKZ Autonomous Agents Integration
 * OJSkin - Open Journal Systems with Skin Zone Journal
 * 
 * Provides a clean async API for interacting with the Web Worker.
 * Zero tolerance for mock implementations - production quality only.
 */

// ============================================================================
// Chat Templates
// ============================================================================

const TEMPLATES = {
    chatml: {
        system: '<|im_start|>system\n{content}<|im_end|>\n',
        user: '<|im_start|>user\n{content}<|im_end|>\n',
        assistant: '<|im_start|>assistant\n{content}<|im_end|>\n',
        generation: '<|im_start|>assistant\n',
        stop: ['<|im_end|>', '<|im_start|>']
    },
    llama2: {
        system: '<<SYS>>\n{content}\n<</SYS>>\n\n',
        user: '[INST] {content} [/INST]',
        assistant: ' {content} </s><s>',
        generation: '',
        stop: ['</s>']
    },
    llama3: {
        system: '<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
        user: '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>',
        assistant: '<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>',
        generation: '<|start_header_id|>assistant<|end_header_id|>\n\n',
        stop: ['<|eot_id|>', '<|start_header_id|>']
    },
    phi: {
        system: '',
        user: 'Instruct: {content}\n',
        assistant: 'Output: {content}\n',
        generation: 'Output:',
        stop: ['Instruct:', '\n\n']
    },
    alpaca: {
        system: '{content}\n\n',
        user: '### Instruction:\n{content}\n\n',
        assistant: '### Response:\n{content}\n\n',
        generation: '### Response:\n',
        stop: ['###', '\n\n\n']
    }
};

// SKZ Task Types (must match C enum)
const SKZ_TASK = {
    GENERAL: 0,
    RESEARCH: 1,
    SUBMISSION: 2,
    EDITORIAL: 3,
    REVIEW: 4,
    QUALITY: 5,
    PUBLISHING: 6,
    ANALYTICS: 7,
    CLASSIFICATION: 8,
    SUMMARIZATION: 9,
    EXTRACTION: 10
};

// ============================================================================
// Template Utilities
// ============================================================================

function detectTemplate(arch) {
    const archLower = (arch || '').toLowerCase();
    if (archLower.includes('qwen')) return 'chatml';
    if (archLower.includes('llama') && archLower.includes('3')) return 'llama3';
    if (archLower.includes('llama')) return 'llama2';
    if (archLower.includes('phi')) return 'phi';
    if (archLower.includes('mistral')) return 'llama2';
    return 'chatml';
}

function applyTemplate(messages, templateName, addGenerationPrompt = true) {
    const tmpl = TEMPLATES[templateName] || TEMPLATES.chatml;
    let result = '';
    
    for (const msg of messages) {
        const format = tmpl[msg.role];
        if (format) {
            result += format.replace('{content}', msg.content);
        }
    }
    
    if (addGenerationPrompt) {
        result += tmpl.generation;
    }
    
    return result;
}

// ============================================================================
// LlamaChat Class
// ============================================================================

/**
 * LlamaChat - High-level chat interface for SKZ agents
 */
export class LlamaChat {
    constructor(options = {}) {
        this.options = {
            workerPath: './worker.js',
            temperature: 0.7,
            topP: 0.9,
            topK: 40,
            maxTokens: 256,
            repeatPenalty: 1.1,
            systemPrompt: 'You are a helpful assistant.',
            template: 'auto',
            ...options
        };
        
        this.worker = null;
        this.callbacks = new Map();
        this.nextId = 0;
        this.modelInfo = null;
        this.messages = [];
        this.ready = false;
        this.loading = false;
    }
    
    /**
     * Initialize the worker and WASM module.
     */
    async init() {
        return new Promise((resolve, reject) => {
            this.worker = new Worker(this.options.workerPath);
            
            this.worker.onmessage = (e) => {
                const { type, id, data } = e.data;
                
                // Handle worker ready
                if (type === 'worker_ready') {
                    this._call('init').then(resolve).catch(reject);
                    return;
                }
                
                // Handle log messages
                if (type === 'log') {
                    console.log(`[LlamaChat] ${data.level}: ${data}`);
                    return;
                }
                
                // Handle callbacks
                const callback = this.callbacks.get(id);
                if (callback) {
                    callback(type, data);
                }
            };
            
            this.worker.onerror = (e) => {
                reject(new Error(`Worker error: ${e.message}`));
            };
        });
    }
    
    /**
     * Load a model from URL or ArrayBuffer.
     */
    async loadModel(model, options = {}) {
        const { contextSize = 0, onProgress, onDownloadProgress } = options;
        
        this.loading = true;
        
        return new Promise((resolve, reject) => {
            const id = this.nextId++;
            
            this.callbacks.set(id, (type, data) => {
                switch (type) {
                    case 'download_progress':
                        if (onDownloadProgress) onDownloadProgress(data);
                        break;
                    case 'progress':
                        if (onProgress) onProgress(data);
                        break;
                    case 'status':
                        console.log(`[LlamaChat] ${data}`);
                        break;
                    case 'loaded':
                        this.callbacks.delete(id);
                        this.modelInfo = data.info;
                        this.ready = true;
                        this.loading = false;
                        
                        // Auto-detect template
                        if (this.options.template === 'auto') {
                            this.options.template = detectTemplate(this.modelInfo?.arch);
                        }
                        
                        resolve(data);
                        break;
                    case 'error':
                        this.callbacks.delete(id);
                        this.loading = false;
                        reject(new Error(data));
                        break;
                }
            });
            
            this.worker.postMessage({
                type: 'load',
                id,
                data: { model, contextSize }
            });
        });
    }
    
    /**
     * Send a message and get a response (blocking).
     */
    async send(content, options = {}) {
        const tokens = [];
        
        for await (const token of this.stream(content, options)) {
            tokens.push(token);
        }
        
        return tokens.join('');
    }
    
    /**
     * Send a message and stream the response.
     */
    async *stream(content, options = {}) {
        if (!this.ready) {
            throw new Error('Model not loaded');
        }
        
        const params = {
            maxTokens: options.maxTokens ?? this.options.maxTokens,
            temperature: options.temperature ?? this.options.temperature,
            topP: options.topP ?? this.options.topP,
            topK: options.topK ?? this.options.topK,
            repeatPenalty: options.repeatPenalty ?? this.options.repeatPenalty,
            stream: true
        };
        
        // Add user message
        this.messages.push({ role: 'user', content });
        
        // Build prompt with chat template
        const allMessages = [];
        if (this.options.systemPrompt) {
            allMessages.push({ role: 'system', content: this.options.systemPrompt });
        }
        allMessages.push(...this.messages);
        
        const prompt = applyTemplate(allMessages, this.options.template, true);
        
        // Generate
        const tokens = [];
        const id = this.nextId++;
        
        const result = await new Promise((resolve, reject) => {
            this.callbacks.set(id, (type, data) => {
                switch (type) {
                    case 'token':
                        tokens.push(data.text);
                        break;
                    case 'done':
                        this.callbacks.delete(id);
                        resolve(data);
                        break;
                    case 'error':
                        this.callbacks.delete(id);
                        reject(new Error(data));
                        break;
                }
            });
            
            this.worker.postMessage({
                type: 'generate',
                id,
                data: { prompt, params }
            });
        });
        
        // Yield tokens
        for (const token of tokens) {
            yield token;
        }
        
        // Add assistant response to history
        const response = tokens.join('');
        this.messages.push({ role: 'assistant', content: response });
        
        return result;
    }
    
    /**
     * Clear conversation history.
     */
    async clear() {
        this.messages = [];
        return this._call('clear');
    }
    
    /**
     * Abort ongoing generation.
     */
    abort() {
        this.worker.postMessage({ type: 'abort', id: -1 });
    }
    
    /**
     * Get model information.
     */
    getInfo() {
        return this.modelInfo;
    }
    
    /**
     * Get conversation history.
     */
    getHistory() {
        return [...this.messages];
    }
    
    /**
     * Set system prompt.
     */
    setSystemPrompt(prompt) {
        this.options.systemPrompt = prompt;
    }
    
    /**
     * Get generation statistics.
     */
    async getStats() {
        return this._call('stats');
    }
    
    /**
     * Get memory usage information.
     */
    async getMemoryInfo() {
        return this._call('memory');
    }
    
    /**
     * Internal: call worker and wait for response.
     */
    _call(type, data = {}) {
        return new Promise((resolve, reject) => {
            const id = this.nextId++;
            
            this.callbacks.set(id, (respType, respData) => {
                this.callbacks.delete(id);
                if (respType === 'error') {
                    reject(new Error(respData));
                } else {
                    resolve(respData);
                }
            });
            
            this.worker.postMessage({ type, id, data });
        });
    }
    
    /**
     * Terminate worker.
     */
    terminate() {
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }
        this.ready = false;
    }
}

// ============================================================================
// SKZAgent Class - Specialized for SKZ Autonomous Agents
// ============================================================================

/**
 * SKZAgent - SKZ Autonomous Agent interface
 */
export class SKZAgent extends LlamaChat {
    constructor(taskType, options = {}) {
        const systemPrompts = {
            [SKZ_TASK.RESEARCH]: 'You are a research discovery agent specializing in academic literature analysis for cosmetic science.',
            [SKZ_TASK.SUBMISSION]: 'You are a submission assistant agent helping authors prepare manuscripts for academic journals.',
            [SKZ_TASK.EDITORIAL]: 'You are an editorial orchestration agent managing publication workflows efficiently.',
            [SKZ_TASK.REVIEW]: 'You are a review coordination agent matching reviewers to manuscripts based on expertise.',
            [SKZ_TASK.QUALITY]: 'You are a content quality agent ensuring publication standards and scientific rigor.',
            [SKZ_TASK.PUBLISHING]: 'You are a publishing production agent handling final publication steps.',
            [SKZ_TASK.ANALYTICS]: 'You are an analytics agent providing insights on publication metrics and trends.',
            [SKZ_TASK.CLASSIFICATION]: 'You are a text classification agent. Classify inputs accurately.',
            [SKZ_TASK.SUMMARIZATION]: 'You are a summarization agent. Provide concise, accurate summaries.',
            [SKZ_TASK.EXTRACTION]: 'You are an information extraction agent. Extract structured data from text.'
        };
        
        super({
            systemPrompt: systemPrompts[taskType] || 'You are a helpful AI assistant.',
            ...options
        });
        
        this.taskType = taskType;
    }
    
    /**
     * Process input using SKZ-optimized generation.
     */
    async process(input, options = {}) {
        if (!this.ready) {
            throw new Error('Model not loaded');
        }
        
        return new Promise((resolve, reject) => {
            const id = this.nextId++;
            
            this.callbacks.set(id, (type, data) => {
                this.callbacks.delete(id);
                if (type === 'error') {
                    reject(new Error(data));
                } else {
                    resolve(data.text);
                }
            });
            
            this.worker.postMessage({
                type: 'skz_generate',
                id,
                data: {
                    input,
                    taskType: this.taskType,
                    params: {
                        maxTokens: options.maxTokens ?? this.options.maxTokens,
                        temperature: options.temperature ?? this.options.temperature,
                        topP: options.topP ?? this.options.topP,
                        topK: options.topK ?? this.options.topK,
                        repeatPenalty: options.repeatPenalty ?? this.options.repeatPenalty
                    }
                }
            });
        });
    }
    
    /**
     * Classify text into categories.
     */
    async classify(text, categories) {
        if (!this.ready) {
            throw new Error('Model not loaded');
        }
        
        return new Promise((resolve, reject) => {
            const id = this.nextId++;
            
            this.callbacks.set(id, (type, data) => {
                this.callbacks.delete(id);
                if (type === 'error') {
                    reject(new Error(data));
                } else {
                    resolve(data.classification);
                }
            });
            
            this.worker.postMessage({
                type: 'skz_classify',
                id,
                data: { text, categories }
            });
        });
    }
    
    /**
     * Extract structured information from text.
     */
    async extract(text, schema) {
        if (!this.ready) {
            throw new Error('Model not loaded');
        }
        
        return new Promise((resolve, reject) => {
            const id = this.nextId++;
            
            this.callbacks.set(id, (type, data) => {
                this.callbacks.delete(id);
                if (type === 'error') {
                    reject(new Error(data));
                } else {
                    try {
                        resolve(JSON.parse(data.extracted));
                    } catch {
                        resolve(data.extracted);
                    }
                }
            });
            
            this.worker.postMessage({
                type: 'skz_extract',
                id,
                data: { text, schema: JSON.stringify(schema) }
            });
        });
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Simple one-shot generation (no chat history).
 */
export async function generate(model, prompt, options = {}) {
    const chat = new LlamaChat({
        systemPrompt: '',
        ...options
    });
    
    await chat.init();
    await chat.loadModel(model);
    
    const response = await chat.send(prompt, options);
    chat.terminate();
    
    return response;
}

/**
 * Create an SKZ agent for a specific task.
 */
export function createAgent(taskType, options = {}) {
    return new SKZAgent(taskType, options);
}

// ============================================================================
// Exports
// ============================================================================

export {
    TEMPLATES,
    SKZ_TASK,
    applyTemplate,
    detectTemplate
};
