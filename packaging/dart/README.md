# edge_intelligence

Dart bindings for the Edge Intelligence SDK, including Flutter Android/iOS and
native Dart desktop runtimes.

The package exposes one framework-neutral Dart API. Flutter plugin metadata is
used only to bundle the Android and iOS native runtimes; the public library does
not import Flutter or expose Flutter types.

## Usage

```dart
import "package:edge_intelligence/edge_intelligence.dart";

Future<void> main() async {
  await initEdgeIntelligence();

  try {
    final sdk = await EdgeLlm.local("/path/to/model.gguf");
    final reply = await sdk.ask("Summarize edge inference in one sentence.");
    print(reply);
  } finally {
    disposeEdgeIntelligence();
  }
}
```

On a desktop host, `initEdgeIntelligence` resolves the packaged artifact for the
current operating system:

- Linux x64: `lib/native/linux/x64/libel_ffi.so`
- macOS universal: `lib/native/macos/libel_ffi.dylib`
- Windows x64: `lib/native/windows/x64/el_ffi.dll`

If the package assets cannot be loaded from the host filesystem,
initialization fails with an actionable `UnsupportedError`; hosts with a custom
runtime location can pass an `externalLibrary` directly.

## Flutter Android and iOS

Add `edge_intelligence` as a normal Flutter dependency. The plugin bundles
`libel_ffi.so` on Android and `el_ffi.xcframework` on iOS, so no manual native
library path is required.

```dart
import "package:edge_intelligence/edge_intelligence.dart";
import "package:flutter/widgets.dart";

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initEdgeIntelligence();
  runApp(const MyApp());
}
```

Call `disposeEdgeIntelligence()` from the root application's teardown hook, not
immediately after `runApp`, because `runApp` returns while the application is
still active.

The mobile release includes:

- Android API 24+ for `armeabi-v7a`, `arm64-v8a`, and `x86_64`.
- iOS 13+ device and simulator slices in a dynamic XCFramework.

Keep framework-specific model storage, permissions, and UI state in the Flutter
application. Pass the resulting local GGUF file path to `EdgeLlm.local`.

## Supported hosts

| Host | Runtime packaging |
|---|---|
| Flutter Android | APK/AAB native libraries |
| Flutter iOS | SwiftPM or CocoaPods XCFramework |
| Dart Linux x64 | Packaged shared library |
| Dart macOS | Packaged universal dynamic library |
| Dart Windows x64 | Packaged DLL |

Web is not currently supported by this pub.dev package. Use the npm/WASM package
for browser applications.

## Example

`example/` contains a runnable Flutter Android/iOS application that selects a
local GGUF model and streams a response. Run it with:

```shell
cd example
flutter pub get
flutter run
```

`example/dart_cli.dart` demonstrates the same API from a Dart entrypoint. Because
this package includes Flutter plugin metadata, dependency resolution requires a
Flutter SDK even when the consuming code imports no Flutter APIs.
