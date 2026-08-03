Release assembly places `el_ffi.xcframework` here. Swift Package Manager
requires the local binary target to resolve inside the package root. CocoaPods
resolves `vendored_frameworks` relative to `ios/edge_intelligence.podspec`,
which points to this same framework through `edge_intelligence/Frameworks/`.
