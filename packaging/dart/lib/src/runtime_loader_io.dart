import 'dart:ffi' show Abi;
import 'dart:io';
import 'dart:isolate';

import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart'
    show ExternalLibrary;

Future<ExternalLibrary> loadPackagedExternalLibrary() async {
  if (Platform.isAndroid) {
    return ExternalLibrary.open(
      'libel_ffi.so',
      debugInfo: 'Flutter Android FFI plugin',
    );
  }
  if (Platform.isIOS) {
    return ExternalLibrary.open(
      'el_ffi.framework/el_ffi',
      debugInfo: 'Flutter iOS FFI plugin',
    );
  }

  final libraryPath = _platformLibraryPath();
  final packageUri = await Isolate.resolvePackageUri(
    Uri.parse('package:edge_intelligence/native/$libraryPath'),
  );
  if (packageUri == null) {
    throw UnsupportedError(
      'edge_intelligence: cannot locate packaged native library '
      '($libraryPath). This host does not expose package assets as files. '
      'Pass externalLibrary explicitly.',
    );
  }
  if (!packageUri.isScheme('file')) {
    throw UnsupportedError(
      'edge_intelligence: package URI resolved to a non-file scheme '
      '(${packageUri.scheme}); cannot load packaged native library. '
      'Pass externalLibrary explicitly.',
    );
  }

  final file = File.fromUri(packageUri);
  if (!await file.exists()) {
    throw UnsupportedError(
      'edge_intelligence: packaged native library is missing at '
      '${file.path}. Reinstall the package or pass externalLibrary explicitly.',
    );
  }

  return ExternalLibrary.open(file.path, debugInfo: packageUri.toString());
}

String _platformLibraryPath() {
  final abi = Abi.current();
  if (Platform.isWindows) {
    if (abi == Abi.windowsX64) {
      return 'windows/x64/el_ffi.dll';
    }
    throw UnsupportedError(_unsupportedRuntimeMessage());
  }
  if (Platform.isMacOS) {
    return 'macos/libel_ffi.dylib';
  }
  if (Platform.isLinux) {
    if (abi == Abi.linuxX64) {
      return 'linux/x64/libel_ffi.so';
    }
    throw UnsupportedError(_unsupportedRuntimeMessage());
  }
  throw UnsupportedError(_unsupportedRuntimeMessage());
}

String _unsupportedRuntimeMessage() =>
    'edge_intelligence does not currently ship a pub.dev native runtime for '
    '${Platform.operatingSystem}/${Abi.current()}. Use a host-specific adapter '
    'or pass externalLibrary explicitly.';
