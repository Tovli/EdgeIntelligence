import type { EdgeLlm, EdgeLlmLike } from './el_ffi';

type NativeBindings = typeof import('./native');

function loadNativeBindings(): NativeBindings {
  try {
    // UBRN's generated entrypoint obtains the TurboModule, installs the Rust
    // crate into Hermes, and initializes the generated UniFFI bindings.
    const bindings = require('./native') as NativeBindings;
    if (
      !bindings.EdgeLlm
      || typeof bindings.EdgeLlm.local !== 'function'
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

export function localEdgeLlm(modelUri: string): EdgeLlmLike {
  const native = loadNativeBindings();
  return native.EdgeLlm.local(modelUri);
}

export function cloudEdgeLlm(model: string, apiKey: string): EdgeLlmLike {
  const native = loadNativeBindings();
  return native.EdgeLlm.cloud(model, apiKey);
}

export type { EdgeLlm, EdgeLlmLike, SdkError } from './el_ffi';
