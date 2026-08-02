import type { EdgeLlm, EdgeLlmLike, SdkError } from './el_ffi';

export declare function requireNativeElFfi(): void;
export declare function localEdgeLlm(modelUri: string): EdgeLlmLike;
export declare function cloudEdgeLlm(model: string, apiKey: string): EdgeLlmLike;

export type { EdgeLlm, EdgeLlmLike, SdkError };
