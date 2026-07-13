import 'dart:io';

import 'package:edge_intelligence/edge_intelligence.dart';

Future<void> main(List<String> arguments) async {
  if (arguments.isEmpty) {
    stderr
        .writeln('Usage: dart run example/dart_cli.dart <model.gguf> [prompt]');
    exitCode = 64;
    return;
  }

  final prompt = arguments.length > 1
      ? arguments.skip(1).join(' ')
      : 'Summarize edge inference in one sentence.';

  await initEdgeIntelligence();
  try {
    final sdk = await EdgeLlm.local(arguments.first);
    await for (final token in sdk.askStream(prompt)) {
      stdout.write(token);
    }
    stdout.writeln();
  } finally {
    disposeEdgeIntelligence();
  }
}
