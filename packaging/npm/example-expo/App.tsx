import { askAsync, askStreamAsync, localEdgeLlm } from 'edge-intelligence-sdk';
import { useEffect, useState } from 'react';
import { Text, View } from 'react-native';

// CI copies the caller-provided Qwen assets into this app-specific external
// directory before launch. Android grants the app access to its own external
// files directory without a storage permission; production callers must make
// the same choice instead of relying on a package-bundled model.
const modelUri = '/storage/emulated/0/Android/data/com.tovli.edgeintelligence.example/files/models/qwen.gguf';
const tokenizerUri = '/storage/emulated/0/Android/data/com.tovli.edgeintelligence.example/files/models/tokenizer.json';
const successText = 'Edge Intelligence Qwen local session passed.';
const prompt = 'Reply with exactly: ready';
const resetDeadlineMs = 30_000;

function describeError(error: unknown): string {
  const fallback = String(error);
  if (
    typeof error !== 'object'
    || error === null
    || !('inner' in error)
    || typeof error.inner !== 'object'
    || error.inner === null
    || !('message' in error.inner)
    || typeof error.inner.message !== 'string'
  ) {
    return fallback;
  }

  return `${fallback}: ${error.inner.message}`;
}

function assertReady(label: string, response: string): void {
  const normalized = response.trim().toLowerCase();
  if (
    normalized.length === 0
    || normalized.split('').every((character) => character === '?')
    || normalized === "i can't help with that request."
    || !normalized.includes('ready')
  ) {
    throw new Error(`${label} did not contain a decoded ready response: ${response}`);
  }
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}

async function expectRejection(
  promise: Promise<unknown>,
  expectedMessage: RegExp,
  label: string,
): Promise<void> {
  let rejected = false;
  let message = '';
  try {
    await promise;
  } catch (error) {
    rejected = true;
    message = describeError(error);
  }

  if (!rejected || !expectedMessage.test(message)) {
    throw new Error(`${label} expected ${expectedMessage}, received: ${message || 'a resolution'}`);
  }
}

async function resetWhenIdle(sdk: ReturnType<typeof localEdgeLlm>): Promise<void> {
  const deadline = Date.now() + resetDeadlineMs;
  for (;;) {
    try {
      sdk.reset();
      return;
    } catch (error) {
      if (!/busy/i.test(describeError(error)) || Date.now() >= deadline) {
        throw error;
      }
      await delay(25);
    }
  }
}

async function assertImmediateCancellation(sdk: ReturnType<typeof localEdgeLlm>): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    let terminalCalls = 0;
    const request = askStreamAsync(sdk, prompt, {
      onToken(token) {
        reject(new Error(`askStreamAsync delivered a token after immediate cancellation: ${token}`));
      },
      onComplete() {
        terminalCalls += 1;
        reject(new Error('askStreamAsync completed after immediate cancellation'));
      },
      onError(error) {
        terminalCalls += 1;
        reject(new Error(`askStreamAsync errored after immediate cancellation: ${error}`));
      },
      onCancelled() {
        terminalCalls += 1;
        if (terminalCalls !== 1) {
          reject(new Error('askStreamAsync delivered more than one terminal callback'));
          return;
        }
        setTimeout(() => {
          if (terminalCalls === 1) {
            resolve();
          } else {
            reject(new Error('askStreamAsync delivered more than one terminal callback'));
          }
        }, 25);
      },
    });
    request.cancel();
  });
}

async function assertImmediateCompletionCancellation(
  sdk: ReturnType<typeof localEdgeLlm>,
): Promise<void> {
  const completion = askAsync(sdk, prompt);
  completion.request.cancel();
  await expectRejection(
    completion.response,
    /request cancelled/i,
    'immediately cancelled askAsync',
  );
}

async function runQwenLocalSessionSmoke(): Promise<string> {
  const sdk = localEdgeLlm(modelUri, tokenizerUri);

  const completion = askAsync(sdk, prompt);
  let completionSettled = false;
  void completion.response.then(
    () => { completionSettled = true; },
    () => { completionSettled = true; },
  );
  const overlap = askAsync(sdk, prompt);
  await expectRejection(overlap.response, /busy/i, 'overlapping askAsync');
  await new Promise<void>((resolve, reject) => {
    setTimeout(() => {
      if (completionSettled) {
        reject(new Error('askAsync settled before the JavaScript event loop progressed'));
      } else {
        resolve();
      }
    }, 0);
  });
  const reply = await completion.response;
  assertReady('askAsync', reply);
  await resetWhenIdle(sdk);

  await assertImmediateCompletionCancellation(sdk);
  await resetWhenIdle(sdk);

  await assertImmediateCancellation(sdk);
  await resetWhenIdle(sdk);

  const streamed = await new Promise<string>((resolve, reject) => {
    let tokens = '';
    askStreamAsync(sdk, prompt, {
      onToken(token) {
        tokens += token;
      },
      onComplete() {
        resolve(tokens);
      },
      onError(error) {
        reject(new Error(error));
      },
      onCancelled() {
        reject(new Error('askStreamAsync was cancelled'));
      },
    });
  });
  assertReady('askStreamAsync', streamed);
  await resetWhenIdle(sdk);

  return successText;
}

export default function App() {
  const [resultText, setResultText] = useState('Edge Intelligence Qwen local session running.');

  useEffect(() => {
    let mounted = true;
    // This device/emulator fixture proves non-blocking submission, immediate
    // stateful Busy rejection, cancellation terminal ordering, and the UniFFI
    // CallInvoker-backed stream callbacks against the production Qwen provider.
    void runQwenLocalSessionSmoke()
      .then((result) => {
        if (mounted) {
          setResultText(result);
        }
      })
      .catch((error) => {
        if (mounted) {
          setResultText(`Edge Intelligence Qwen local session failed: ${describeError(error)}`);
        }
      });

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <View>
      <Text>{resultText}</Text>
    </View>
  );
}
