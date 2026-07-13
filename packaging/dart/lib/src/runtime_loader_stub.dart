import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart'
    show ExternalLibrary;

Future<ExternalLibrary?> loadPackagedExternalLibrary() async {
  throw UnsupportedError(
    'edge_intelligence pub.dev does not support this runtime environment. '
    'Expected a native desktop (dart.library.io) or web (dart.library.js_interop) '
    'context. Pass externalLibrary explicitly if your platform is supported.',
  );
}
