# Edge Intelligence Flutter example

This Android/iOS example initializes the packaged Rust runtime, lets the user
select a local GGUF model, and streams a response through the public Dart API.
It uses the retained byte-level compatibility path, which is **not compatible
with Qwen2/Qwen2.5 chat models**. Do not select a Qwen GGUF: the Dart binding
does not yet accept its required `tokenizer.json`.

```shell
flutter pub get
flutter run
```
