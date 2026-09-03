import type { EdgeLlm, EdgeLlmLike, SdkError } from './el_ffi';

export declare function requireNativeElFfi(): void;
export interface AsyncRequest {
  cancel(): void;
  isCancelled(): boolean;
}
export interface AsyncStreamHandler {
  onToken(token: string): void;
  onComplete(): void;
  onError(error: string): void;
  onCancelled(): void;
}
export interface AsyncCompletion {
  request: AsyncRequest;
  response: Promise<string>;
}
/** Native submission and binding-version failures reject `response`; never throws synchronously. */
export declare function askAsync(sdk: EdgeLlmLike, prompt: string): AsyncCompletion;
/** Submission and binding-version failures are delivered asynchronously to `onError`. */
export declare function askStreamAsync(
  sdk: EdgeLlmLike,
  prompt: string,
  handler: AsyncStreamHandler,
): AsyncRequest;
/** @deprecated Pass a matching tokenizer: `localEdgeLlm(modelUri, tokenizerUri)`. */
export declare function localEdgeLlm(modelUri: string): EdgeLlmLike;
export declare function localEdgeLlm(modelUri: string, tokenizerUri: string): EdgeLlmLike;
export declare function cloudEdgeLlm(model: string, apiKey: string): EdgeLlmLike;

export type { EdgeLlm, EdgeLlmLike, SdkError };
