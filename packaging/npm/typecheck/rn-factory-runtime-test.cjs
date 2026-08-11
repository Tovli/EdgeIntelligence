'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const typescript = require('typescript');

const entrypointPath = path.join(__dirname, '..', 'src', 'rn', 'index.ts');
const source = fs.readFileSync(entrypointPath, 'utf8');
const transpiled = typescript.transpileModule(source, {
  compilerOptions: {
    module: typescript.ModuleKind.CommonJS,
    target: typescript.ScriptTarget.ES2020,
  },
  fileName: entrypointPath,
  reportDiagnostics: true,
});

const diagnostics = (transpiled.diagnostics || []).filter(
  (diagnostic) => diagnostic.category === typescript.DiagnosticCategory.Error,
);
assert.deepEqual(diagnostics, [], 'React Native factory entrypoint must transpile');

function loadLocalFactory(nativeBindings) {
  const module = { exports: {} };
  const localRequire = (specifier) => {
    if (specifier === './native') {
      return nativeBindings;
    }
    throw new Error(`unexpected runtime import: ${specifier}`);
  };

  vm.runInNewContext(
    transpiled.outputText,
    { Error, exports: module.exports, module, require: localRequire },
    { filename: entrypointPath },
  );
  return module.exports.localEdgeLlm;
}

const localQwenCalls = [];
const expectedSession = { ask: () => 'ready' };
const localEdgeLlm = loadLocalFactory({
  EdgeLlm: {
    cloud: () => ({ ask: () => 'cloud' }),
    localQwen: (modelUri, tokenizerUri) => {
      localQwenCalls.push([modelUri, tokenizerUri]);
      return expectedSession;
    },
  },
});

assert.throws(
  () => localEdgeLlm('', undefined),
  /modelUri must be a non-empty local file path/,
  'model validation must take precedence over the legacy-overload migration error',
);
assert.throws(
  () => localEdgeLlm('/models/qwen.gguf'),
  /Migrate to localEdgeLlm\(modelUri, tokenizerUri\)/,
  'the retained one-argument overload must direct callers to the tokenizer-aware API',
);
assert.throws(
  () => localEdgeLlm('/models/qwen.gguf', ''),
  /tokenizerUri must be a non-empty local file path/,
  'the two-argument overload must validate the tokenizer path before native loading',
);
assert.deepEqual(localQwenCalls, [], 'invalid calls must not construct a native session');

assert.strictEqual(
  localEdgeLlm('/models/qwen.gguf', '/models/tokenizer.json'),
  expectedSession,
  'the two-argument overload must return the generated native Qwen session',
);
assert.deepEqual(localQwenCalls, [
  ['/models/qwen.gguf', '/models/tokenizer.json'],
]);

console.log('React Native localEdgeLlm factory behavior: OK');
