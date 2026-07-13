import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart'
    show ExternalLibrary;

Future<ExternalLibrary?> loadPackagedExternalLibrary() async {
  throw UnsupportedError(
    'edge_intelligence pub.dev does not currently ship a web/WASM '
    'flutter_rust_bridge runtime. Use the npm/web package for browser hosts.',
  );
}
