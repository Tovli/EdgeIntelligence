import type { EdgeLlm, EdgeLlmLike } from './el_ffi';

type NativeBindings = typeof import('./native');
declare const require: (moduleId: string) => unknown;

function loadNativeBindings(): NativeBindings {
  try {
    // UBRN's generated entrypoint obtains the TurboModule, installs the Rust
    // crate into Hermes, and initializes the generated UniFFI bindings.
    const bindings = require('./native') as NativeBindings;
    if (
      !bindings.EdgeLlm
      || typeof bindings.EdgeLlm.localQwen !== 'function'
      || typeof bindings.EdgeLlm.cloud !== 'function'
    ) {
      throw new Error('generated native entrypoint did not export EdgeLlm');
    }
    return bindings;
  } catch (cause) {
    const detail = cause instanceof Error ? ` Original error: ${cause.message}` : '';
    throw new Error(
      'edge-intelligence-sdk native module is unavailable. Build or rebuild a '
        + 'React Native app (or an Expo development/production build); Expo Go '
        + `cannot load this package.${detail}`
    );
  }
}

export function requireNativeElFfi(): void {
  loadNativeBindings();
}

function requireLocalAssetPath(path: string, label: string): void {
  if (path.trim().length === 0) {
    throw new Error(`${label} must be a non-empty local file path.`);
  }
}

/**
 * @deprecated A Qwen session requires a matching tokenizer. Pass it as the
 * second argument: `localEdgeLlm(modelUri, tokenizerUri)`.
 *
 * This overload is retained for the published 0.3.x signature and, in the
 * 0.4.0-or-later migration release, throws an error before constructing a
 * byte-level session.
 */
export function localEdgeLlm(modelUri: string): EdgeLlmLike;
export function localEdgeLlm(modelUri: string, tokenizerUri: string): EdgeLlmLike;
export function localEdgeLlm(modelUri: string, tokenizerUri?: string): EdgeLlmLike {
  requireLocalAssetPath(modelUri, 'modelUri');
  if (tokenizerUri === undefined) {
    throw new Error(
      'localEdgeLlm(modelUri) requires a matching tokenizerUri for Qwen. '
        + 'Migrate to localEdgeLlm(modelUri, tokenizerUri).'
    );
  }
  requireLocalAssetPath(tokenizerUri, 'tokenizerUri');
  const native = loadNativeBindings();
  return native.EdgeLlm.localQwen(modelUri, tokenizerUri);
}

export function cloudEdgeLlm(model: string, apiKey: string): EdgeLlmLike {
  const native = loadNativeBindings();
  return native.EdgeLlm.cloud(model, apiKey);
}

export type { EdgeLlm, EdgeLlmLike, SdkError } from './el_ffi';
