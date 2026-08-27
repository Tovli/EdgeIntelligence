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

function loadEntrypoint(nativeBindings) {
  const module = { exports: {} };
  const localRequire = (specifier) => {
    if (specifier === './native') {
      return nativeBindings;
    }
    throw new Error(`unexpected runtime import: ${specifier}`);
  };

  vm.runInNewContext(
    transpiled.outputText,
    { Error, Promise, exports: module.exports, module, require: localRequire },
    { filename: entrypointPath },
  );
  return module.exports;
}

async function main() {
  const localQwenCalls = [];
  const expectedSession = { ask: () => 'ready' };
  const entrypoint = loadEntrypoint({
    EdgeLlm: {
      cloud: () => ({ ask: () => 'cloud' }),
      localQwen: (modelUri, tokenizerUri) => {
        localQwenCalls.push([modelUri, tokenizerUri]);
        return expectedSession;
      },
    },
  });
  const { askAsync, askStreamAsync, localEdgeLlm } = entrypoint;

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

  let busyCompletion;
  assert.doesNotThrow(() => {
    busyCompletion = askAsync({
      askAsync() {
        throw new Error('busy');
      },
      askStreamAsync() {},
    }, 'prompt');
  }, 'native Busy must reject the response instead of throwing synchronously');
  assert.doesNotThrow(() => busyCompletion.request.cancel());
  assert.strictEqual(busyCompletion.request.isCancelled(), false);
  await assert.rejects(busyCompletion.response, /busy/);

  let mismatchCompletion;
  assert.doesNotThrow(() => {
    mismatchCompletion = askAsync({}, 'prompt');
  }, 'binding mismatch must reject the response instead of throwing synchronously');
  await assert.rejects(mismatchCompletion.response, /out of sync/);

  let streamSubmissionReturned = false;
  const streamError = new Promise((resolve) => {
    const request = askStreamAsync({
      askAsync() {},
      askStreamAsync() {
        throw new Error('stream busy');
      },
    }, 'prompt', {
      onToken() {},
      onComplete() {},
      onError(message) {
        assert.strictEqual(
          streamSubmissionReturned,
          true,
          'stream submission errors must not invoke onError synchronously',
        );
        resolve(message);
      },
      onCancelled() {},
    });
    assert.doesNotThrow(() => request.cancel());
    assert.strictEqual(request.isCancelled(), false);
  });
  streamSubmissionReturned = true;
  await assert.doesNotReject(streamError.then((message) => {
    assert.match(message, /stream busy/);
  }));

  console.log('React Native factory and async wrapper behavior: OK');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
