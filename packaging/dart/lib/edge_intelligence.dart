/// Dart bindings for the Edge Intelligence SDK.
library;

import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart'
    show ExternalLibrary;

import 'src/dart_api.dart' as dart_api;
import 'src/frb_generated.dart' show RustLib;
import 'src/lib.dart' as generated;
import 'src/runtime_loader.dart' as runtime_loader;

export 'src/frb_generated.dart' show RustLib, RustLibApi, RustLibApiImpl;

/// Edge Intelligence LLM SDK handle.
final class EdgeLlm {
  EdgeLlm._(this._sdk);

  final generated.EdgeLlm _sdk;

  /// Creates a local, air-gapped LLM provider.
  static Future<EdgeLlm> local(String modelUri) async {
    return EdgeLlm._(await dart_api.edgeLlmLocal(modelUri: modelUri));
  }

  /// Creates an opt-in frontier cloud provider.
  static Future<EdgeLlm> cloud(String model, String apiKey) async {
    return EdgeLlm._(await dart_api.edgeLlmCloud(model: model, apiKey: apiKey));
  }

  /// Runs a single prompt and returns the complete response.
  Future<String> ask(String prompt) {
    return dart_api.edgeLlmAsk(sdk: _sdk, prompt: prompt);
  }

  /// Streams response tokens for a prompt.
  ///
  /// If the provider fails after emitting tokens, the returned stream emits a
  /// terminal error event before it closes.
  Stream<String> askStream(String prompt) {
    return dart_api.edgeLlmAskStream(sdk: _sdk, prompt: prompt);
  }

  /// Clears any cached session state.
  ///
  /// The local provider resets automatically at the start of each call, so
  /// this is a no-op for the current backend. Call it as a forward-compatible
  /// signal if your host logic requires an explicit boundary between sessions.
  Future<void> reset() {
    return dart_api.edgeLlmReset(sdk: _sdk);
  }
}

/// Initializes the Edge Intelligence native runtime.
Future<void> initEdgeIntelligence({
  ExternalLibrary? externalLibrary,
  bool forceSameCodegenVersion = true,
}) async {
  try {
    await RustLib.init(
      externalLibrary:
          externalLibrary ?? await runtime_loader.loadPackagedExternalLibrary(),
      forceSameCodegenVersion: forceSameCodegenVersion,
    );
  } catch (error, stackTrace) {
    if (error is UnsupportedError) {
      Error.throwWithStackTrace(error, stackTrace);
    }

    Error.throwWithStackTrace(
      StateError(
        'Failed to initialize edge_intelligence native runtime. Ensure the '
        'el_ffi native library is available to the host process, pass '
        'externalLibrary explicitly, or set '
        'FRB_DART_LOAD_EXTERNAL_LIBRARY_NATIVE_LIB_DIR. Original '
        '${error.runtimeType}: $error',
      ),
      stackTrace,
    );
  }
}

/// Releases the Edge Intelligence native runtime.
void disposeEdgeIntelligence() => RustLib.dispose();
