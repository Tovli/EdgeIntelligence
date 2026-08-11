import type { EdgeLlm, EdgeLlmLike, SdkError } from './el_ffi';

export declare function requireNativeElFfi(): void;
/** @deprecated Pass a matching tokenizer: `localEdgeLlm(modelUri, tokenizerUri)`. */
export declare function localEdgeLlm(modelUri: string): EdgeLlmLike;
export declare function localEdgeLlm(modelUri: string, tokenizerUri: string): EdgeLlmLike;
/** Create the pinned, high-memory DictaLM 3.0 Qwen3 local profile. */
export declare function localDictaLm(
  modelUri: string,
  tokenizerUri: string,
  chatTemplateUri: string,
  manifestSignatureUri: string,
  highEnd: boolean,
  memoryBudgetBytes: number,
): EdgeLlmLike;
export declare function cloudEdgeLlm(model: string, apiKey: string): EdgeLlmLike;

export type { EdgeLlm, EdgeLlmLike, SdkError };
