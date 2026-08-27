import {
  askAsync,
  askStreamAsync,
  cloudEdgeLlm,
  localEdgeLlm,
} from 'edge-intelligence-sdk';
import type {
  AsyncCompletion,
  AsyncRequest,
  AsyncStreamHandler,
  EdgeLlmLike,
} from 'edge-intelligence-sdk';

type IsExact<Left, Right> = (<Type>() => Type extends Left ? 1 : 2) extends
  <Type>() => Type extends Right ? 1 : 2
  ? true
  : false;
type Assert<Condition extends true> = Condition;

const legacyLocal: (modelUri: string) => EdgeLlmLike = localEdgeLlm;
const cloud: (model: string, apiKey: string) => EdgeLlmLike = cloudEdgeLlm;
type LocalQwenParameters = Assert<
  IsExact<Parameters<typeof localEdgeLlm>, [modelUri: string, tokenizerUri: string]>
>;
type LocalReturnType = Assert<
  IsExact<ReturnType<typeof localEdgeLlm>, EdgeLlmLike>
>;
type CloudReturnType = Assert<
  IsExact<ReturnType<typeof cloudEdgeLlm>, EdgeLlmLike>
>;
type AsyncCompletionReturnType = Assert<
  IsExact<ReturnType<typeof askAsync>, AsyncCompletion>
>;
type AsyncStreamReturnType = Assert<
  IsExact<ReturnType<typeof askStreamAsync>, AsyncRequest>
>;

const asyncHandler: AsyncStreamHandler = {
  onToken() {},
  onComplete() {},
  onError() {},
  onCancelled() {},
};

void legacyLocal;
void cloud;
void asyncHandler;

// ADR-026 preserves the published one-path signature as a runtime migration
// error so a 0.3.x patch release cannot break consumers at TypeScript build time.
localEdgeLlm('/models/qwen.gguf');
