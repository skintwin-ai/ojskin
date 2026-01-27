/**
 * test_basic.mjs - Basic tests for llama-tiny-wasm
 * 
 * Run with: node --experimental-wasm-modules tests/test_basic.mjs
 */

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Test utilities
let passed = 0;
let failed = 0;

function test(name, fn) {
    try {
        fn();
        console.log(`✓ ${name}`);
        passed++;
    } catch (error) {
        console.log(`✗ ${name}: ${error.message}`);
        failed++;
    }
}

function assert(condition, message) {
    if (!condition) {
        throw new Error(message || 'Assertion failed');
    }
}

function assertEqual(actual, expected, message) {
    if (actual !== expected) {
        throw new Error(message || `Expected ${expected}, got ${actual}`);
    }
}

// ============================================================================
// Tests
// ============================================================================

console.log('\n=== llama-tiny-wasm Tests ===\n');

// Test 1: Check file structure
test('File structure exists', () => {
    const files = [
        'include/llama-tiny.h',
        'include/gguf.h',
        'src/llama-tiny.c',
        'src/gguf.c',
        'web/worker.js',
        'web/llama-client.js',
        'web/index.html',
        'Makefile',
        'README.md'
    ];
    
    for (const file of files) {
        const path = join(__dirname, '..', file);
        try {
            readFileSync(path);
        } catch {
            throw new Error(`Missing file: ${file}`);
        }
    }
});

// Test 2: Check header defines
test('Header defines version', () => {
    const header = readFileSync(join(__dirname, '../include/llama-tiny.h'), 'utf8');
    assert(header.includes('#define LLAMA_TINY_VERSION "1.0.0"'), 'Missing version define');
    assert(header.includes('#define LLAMA_TINY_VERSION_MAJOR 1'), 'Missing major version');
});

// Test 3: Check API completeness
test('API functions declared', () => {
    const header = readFileSync(join(__dirname, '../include/llama-tiny.h'), 'utf8');
    
    const requiredFunctions = [
        'llama_init',
        'llama_cleanup',
        'llama_load_model',
        'llama_free_model',
        'llama_create_context',
        'llama_free_context',
        'llama_generate',
        'llama_generate_stream',
        'llama_tokenize',
        'llama_detokenize',
        'llama_apply_template',
        'skz_generate',
        'skz_classify',
        'skz_extract'
    ];
    
    for (const fn of requiredFunctions) {
        assert(header.includes(fn), `Missing function: ${fn}`);
    }
});

// Test 4: Check SKZ task types
test('SKZ task types defined', () => {
    const header = readFileSync(join(__dirname, '../include/llama-tiny.h'), 'utf8');
    
    const taskTypes = [
        'SKZ_TASK_GENERAL',
        'SKZ_TASK_RESEARCH',
        'SKZ_TASK_SUBMISSION',
        'SKZ_TASK_EDITORIAL',
        'SKZ_TASK_REVIEW',
        'SKZ_TASK_QUALITY',
        'SKZ_TASK_PUBLISHING',
        'SKZ_TASK_ANALYTICS',
        'SKZ_TASK_CLASSIFICATION',
        'SKZ_TASK_SUMMARIZATION',
        'SKZ_TASK_EXTRACTION'
    ];
    
    for (const task of taskTypes) {
        assert(header.includes(task), `Missing task type: ${task}`);
    }
});

// Test 5: Check chat templates
test('Chat templates defined', () => {
    const header = readFileSync(join(__dirname, '../include/llama-tiny.h'), 'utf8');
    
    const templates = [
        'LLAMA_TEMPLATE_AUTO',
        'LLAMA_TEMPLATE_CHATML',
        'LLAMA_TEMPLATE_LLAMA2',
        'LLAMA_TEMPLATE_LLAMA3',
        'LLAMA_TEMPLATE_PHI',
        'LLAMA_TEMPLATE_ALPACA'
    ];
    
    for (const tmpl of templates) {
        assert(header.includes(tmpl), `Missing template: ${tmpl}`);
    }
});

// Test 6: Check GGUF types
test('GGUF types defined', () => {
    const header = readFileSync(join(__dirname, '../include/gguf.h'), 'utf8');
    
    const types = [
        'GGML_TYPE_F32',
        'GGML_TYPE_F16',
        'GGML_TYPE_Q4_0',
        'GGML_TYPE_Q4_1',
        'GGML_TYPE_Q8_0'
    ];
    
    for (const type of types) {
        assert(header.includes(type), `Missing GGML type: ${type}`);
    }
});

// Test 7: Check worker message types
test('Worker handles all message types', () => {
    const worker = readFileSync(join(__dirname, '../web/worker.js'), 'utf8');
    
    const messageTypes = [
        'init',
        'load',
        'generate',
        'skz_generate',
        'skz_classify',
        'skz_extract',
        'clear',
        'abort',
        'info',
        'stats',
        'memory'
    ];
    
    for (const type of messageTypes) {
        assert(worker.includes(`case '${type}'`), `Missing message handler: ${type}`);
    }
});

// Test 8: Check client exports
test('Client exports correct classes', () => {
    const client = readFileSync(join(__dirname, '../web/llama-client.js'), 'utf8');
    
    assert(client.includes('export class LlamaChat'), 'Missing LlamaChat export');
    assert(client.includes('export class SKZAgent'), 'Missing SKZAgent export');
    assert(client.includes('export {'), 'Missing named exports');
    assert(client.includes('TEMPLATES'), 'Missing TEMPLATES export');
    assert(client.includes('SKZ_TASK'), 'Missing SKZ_TASK export');
});

// Test 9: Check Makefile targets
test('Makefile has required targets', () => {
    const makefile = readFileSync(join(__dirname, '../Makefile'), 'utf8');
    
    const targets = [
        'all:',
        'simd:',
        'debug:',
        'clean:',
        'serve:',
        'test:'
    ];
    
    for (const target of targets) {
        assert(makefile.includes(target), `Missing target: ${target}`);
    }
});

// Test 10: Check implementation completeness
test('Implementation has no TODO markers', () => {
    const files = [
        'src/llama-tiny.c',
        'src/gguf.c',
        'web/worker.js',
        'web/llama-client.js'
    ];
    
    for (const file of files) {
        const content = readFileSync(join(__dirname, '..', file), 'utf8');
        assert(!content.includes('TODO'), `Found TODO in ${file}`);
        assert(!content.includes('FIXME'), `Found FIXME in ${file}`);
        assert(!content.includes('NOT_IMPLEMENTED'), `Found NOT_IMPLEMENTED in ${file}`);
    }
});

// Test 11: Check dark mode default
test('UI defaults to dark mode', () => {
    const html = readFileSync(join(__dirname, '../web/index.html'), 'utf8');
    
    // Should have dark mode CSS variables as default (not in .light-mode)
    assert(html.includes('--bg-primary: #0d1117'), 'Dark mode not default');
    assert(html.includes('.light-mode'), 'Light mode class should exist');
    assert(html.includes('toggleTheme'), 'Theme toggle should exist');
});

// Test 12: Check WASM exports
test('WASM exports defined in Makefile', () => {
    const makefile = readFileSync(join(__dirname, '../Makefile'), 'utf8');
    
    const exports = [
        '_llama_init',
        '_llama_load_model',
        '_llama_generate',
        '_skz_generate',
        '_skz_classify',
        '_skz_extract'
    ];
    
    for (const exp of exports) {
        assert(makefile.includes(`"${exp}"`), `Missing WASM export: ${exp}`);
    }
});

// ============================================================================
// Summary
// ============================================================================

console.log('\n=== Test Summary ===');
console.log(`Passed: ${passed}`);
console.log(`Failed: ${failed}`);
console.log(`Total:  ${passed + failed}`);

if (failed > 0) {
    process.exit(1);
}

console.log('\n✓ All tests passed!\n');
