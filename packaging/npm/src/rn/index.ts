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

/** A cooperative cancellation handle for one accepted native request. */
export interface AsyncRequest {
  cancel(): void;
  isCancelled(): boolean;
}

/** Terminal and token callbacks for a non-blocking native stream. */
export interface AsyncStreamHandler {
  onToken(token: string): void;
  onComplete(): void;
  onError(error: string): void;
  onCancelled(): void;
}

/** A non-blocking completion plus the handle that can cancel it. */
export interface AsyncCompletion {
  request: AsyncRequest;
  response: Promise<string>;
}

type NativeAsyncEdgeLlm = {
  askAsync(prompt: string, handler: {
    onComplete(response: string): void;
    onError(error: string): void;
    onCancelled(): void;
  }): AsyncRequest;
  askStreamAsync(prompt: string, handler: AsyncStreamHandler): AsyncRequest;
};

function inertAsyncRequest(): AsyncRequest {
  return {
    cancel() {},
    isCancelled() {
      return false;
    },
  };
}

function nativeAsync(sdk: EdgeLlmLike): NativeAsyncEdgeLlm {
  const native = sdk as EdgeLlmLike & Partial<NativeAsyncEdgeLlm>;
  if (
    typeof native.askAsync !== 'function'
    || typeof native.askStreamAsync !== 'function'
  ) {
    throw new Error(
      'edge-intelligence-sdk JavaScript and native bindings are out of sync: '
        + 'this native build does not expose ADR-027 async methods. Rebuild the app '
        + 'after upgrading the package.',
    );
  }
  return native as NativeAsyncEdgeLlm;
}

/**
 * Starts inference on SDK-owned native workers.
 *
 * The promise always settles asynchronously. `request.cancel()` requests
 * cooperative cancellation. A stateful turn already active, or this handle's
 * per-handle async capacity, causes the native `Busy` error instead of blocking
 * the JavaScript thread. Stateless providers may accept concurrent calls. A
 * cancellation rejection can arrive before a non-cooperative stateful backend
 * drains; keep that handle unavailable until a later call no longer reports
 * `Busy`. Every native submission or callback-provisioning failure rejects
 * `response`; this function never throws synchronously.
 */
export function askAsync(sdk: EdgeLlmLike, prompt: string): AsyncCompletion {
  let resolve!: (response: string) => void;
  let reject!: (error: unknown) => void;
  const response = new Promise<string>((resolveResponse, rejectResponse) => {
    resolve = resolveResponse;
    reject = rejectResponse;
  });
  // A caller may intentionally retain only `request` in order to cancel and
  // ignore a completion. Mark the original rejection as observed while keeping
  // `response` itself rejectable for callers that do await it.
  void response.catch(() => undefined);
  try {
    const request = nativeAsync(sdk).askAsync(prompt, {
      onComplete: resolve,
      onError: (error) => reject(new Error(error)),
      onCancelled: () => reject(new Error('request cancelled')),
    });
    return { request, response };
  } catch (error) {
    reject(error);
    return { request: inertAsyncRequest(), response };
  }
}

/**
 * Starts a non-blocking native stream and returns its cancellation handle.
 *
 * The UniFFI React Native runtime dispatches callbacks on the JavaScript
 * runtime through React Native's CallInvoker. Keep handlers brief: the native
 * delivery worker waits for each callback to return. Submission and binding
 * mismatches are reported asynchronously through `handler.onError`; this
 * function never throws synchronously.
 */
export function askStreamAsync(
  sdk: EdgeLlmLike,
  prompt: string,
  handler: AsyncStreamHandler,
): AsyncRequest {
  try {
    return nativeAsync(sdk).askStreamAsync(prompt, handler);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    void Promise.resolve().then(() => handler.onError(message));
    return inertAsyncRequest();
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
