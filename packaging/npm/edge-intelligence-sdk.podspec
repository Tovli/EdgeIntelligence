Pod::Spec.new do |s|
  s.name = 'edge-intelligence-sdk'
  s.version = '0.0.0-development'
  s.summary = 'Edge Intelligence UniFFI JSI bridge for React Native'
  s.homepage = 'https://github.com/Tovli/EdgeIntelligence'
  s.license = { :type => 'Apache-2.0' }
  s.authors = { 'Tovli' => 'opensource@tovli.com' }
  minimum_ios_version = respond_to?(:min_ios_version_supported, true) ? min_ios_version_supported : '13.0'
  s.platforms = { :ios => minimum_ios_version }
  s.source = { :git => 'https://github.com/Tovli/EdgeIntelligence.git', :tag => "v#{s.version}" }
  s.source_files = 'src/rn/cpp/**/*.{h,hpp,cpp,mm}',
                   'ios/EdgeIntelligenceSdk.{h,mm}',
                   'ios/generated/**/*.{h,hpp,cpp,m,mm}'
  s.vendored_frameworks = 'ios/el_ffi.xcframework'
  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++20',
    'DEFINES_MODULE' => 'YES'
  }
  s.dependency 'uniffi-bindgen-react-native', '0.31.0-3'

  if respond_to?(:install_modules_dependencies, true)
    install_modules_dependencies(s)
  else
    s.dependency 'React-Core'
    s.dependency 'React-callinvoker'
    s.dependency 'React-jsi'
  end
end
