Pod::Spec.new do |s|
  s.name             = 'edge_intelligence'
  s.version          = '0.1.0'
  s.summary          = 'Flutter mobile runtime for the Edge Intelligence Dart SDK.'
  s.description      = <<-DESC
Precompiled Rust runtime used by the framework-neutral Edge Intelligence Dart API.
                       DESC
  s.homepage         = 'https://github.com/Tovli/EdgeIntelligence'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Tovli' => 'opensource@tovli.com' }
  s.source           = { :path => '.' }
  s.vendored_frameworks = 'Frameworks/el_ffi.xcframework'
  s.dependency 'Flutter'
  s.platform = :ios, '13.0'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }
end
