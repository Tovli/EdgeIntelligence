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

  /// Runs a single prompt off the UI isolate and returns the complete response.
  ///
  /// Stateful local providers permit one active operation per handle. A
  /// concurrent local call fails with `Busy` rather than racing a conversation
  /// session; stateless cloud providers may run concurrent calls.
  Future<String> ask(String prompt) {
    return dart_api.edgeLlmAsk(sdk: _sdk, prompt: prompt);
  }

  /// Streams response tokens for a prompt.
  ///
  /// If the provider fails after emitting tokens, the returned stream emits a
  /// terminal error event before it closes. Cancelling the subscription closes
  /// the native sink, but this current Dart surface does not expose ADR-027's
  /// request handle and does not promise to interrupt inference or networking.
  /// In particular, local Candle and Qwen adapters infer a full reply before
  /// replaying fragments, so cancellation cannot stop that inference.
  Stream<String> askStream(String prompt) {
    return dart_api.edgeLlmAskStream(sdk: _sdk, prompt: prompt);
  }

  /// Clears any cached session state.
  ///
  /// A failure means the provider may retain a stale KV cache. Stop using this
  /// handle or rebuild it rather than starting another conversation. Calling
  /// this while a stateful operation is active fails with `Busy`.
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
