# edge_intelligence

Generated Flutter bindings for the Edge Intelligence SDK.

The package contains Dart bindings generated with flutter_rust_bridge and native
libraries assembled by the release pipeline.

## Usage

Copy the Qwen2.5 0.5B GGUF into app storage, then open the local SDK facade with
that model path.

```dart
import "dart:io";
import "package:edge_intelligence/edge_intelligence.dart";

final qwen05b = "/path/to/app/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";
final sdk = await EdgeLlm.local(qwen05b);

final reply = await sdk.ask("Summarize edge inference in one sentence.");
await for (final token in sdk.askStream("Give me two deployment tips.")) {
  stdout.write(token);
}
```
