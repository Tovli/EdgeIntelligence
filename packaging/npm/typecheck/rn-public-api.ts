import { cloudEdgeLlm, localEdgeLlm } from 'edge-intelligence-sdk';
import type { EdgeLlmLike } from 'edge-intelligence-sdk';

type IsExact<Left, Right> = (<Type>() => Type extends Left ? 1 : 2) extends
  <Type>() => Type extends Right ? 1 : 2
  ? true
  : false;
type Assert<Condition extends true> = Condition;

const local: (modelUri: string) => EdgeLlmLike = localEdgeLlm;
const cloud: (model: string, apiKey: string) => EdgeLlmLike = cloudEdgeLlm;
type LocalReturnType = Assert<
  IsExact<ReturnType<typeof localEdgeLlm>, EdgeLlmLike>
>;
type CloudReturnType = Assert<
  IsExact<ReturnType<typeof cloudEdgeLlm>, EdgeLlmLike>
>;

void local;
void cloud;
