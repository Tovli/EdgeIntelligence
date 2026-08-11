import { localEdgeLlm } from 'edge-intelligence-sdk';
import { Text, View } from 'react-native';

// CI copies the caller-provided Qwen assets into this app-specific external
// directory before launch. Android grants the app access to its own external
// files directory without a storage permission; production callers must make
// the same choice instead of relying on a package-bundled model.
const modelUri = '/storage/emulated/0/Android/data/com.tovli.edgeintelligence.example/files/models/qwen.gguf';
const tokenizerUri = '/storage/emulated/0/Android/data/com.tovli.edgeintelligence.example/files/models/tokenizer.json';
const successText = 'Edge Intelligence Qwen local session passed.';

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

function runQwenLocalSessionSmoke(): string {
  const sdk = localEdgeLlm(modelUri, tokenizerUri);

  const reply = sdk.ask('Reply with exactly: ready');
  assertReady('ask', reply);
  sdk.reset();

  let streamed = '';
  sdk.askStreamCb('Reply with exactly: ready', {
    onToken(token) {
      streamed += token;
    },
  });
  assertReady('askStreamCb', streamed);

  return successText;
}

// The API currently executes synchronously. This fixture intentionally performs
// the bounded, two-request Qwen round trip before rendering so release CI can
// prove the generated React Native constructor, reset, and callback reach the
// production provider.
const resultText = (() => {
  try {
    return runQwenLocalSessionSmoke();
  } catch (error) {
    return `Edge Intelligence Qwen local session failed: ${describeError(error)}`;
  }
})();

export default function App() {
  return (
    <View>
      <Text>{resultText}</Text>
    </View>
  );
}
