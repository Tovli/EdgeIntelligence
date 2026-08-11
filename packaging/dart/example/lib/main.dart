import 'package:edge_intelligence/edge_intelligence.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  Object? initializationError;
  try {
    await initEdgeIntelligence();
    debugPrint('edge_intelligence runtime initialized');
  } catch (error) {
    initializationError = error;
    debugPrint('edge_intelligence runtime initialization failed: $error');
  }

  runApp(EdgeIntelligenceExample(initializationError: initializationError));
}

class EdgeIntelligenceExample extends StatefulWidget {
  const EdgeIntelligenceExample({super.key, this.initializationError});

  final Object? initializationError;

  @override
  State<EdgeIntelligenceExample> createState() =>
      _EdgeIntelligenceExampleState();
}

class _EdgeIntelligenceExampleState extends State<EdgeIntelligenceExample> {
  final _promptController = TextEditingController(
    text: 'Summarize edge inference in one sentence.',
  );

  EdgeLlm? _sdk;
  String? _modelName;
  String _output = '';
  Object? _error;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _error = widget.initializationError;
  }

  @override
  void dispose() {
    _promptController.dispose();
    if (widget.initializationError == null) {
      disposeEdgeIntelligence();
    }
    super.dispose();
  }

  Future<void> _selectModel() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: const ['gguf'],
    );
    final file = result?.files.single;
    if (file?.path == null) {
      return;
    }

    setState(() {
      _busy = true;
      _error = null;
      _output = '';
    });

    try {
      final sdk = await EdgeLlm.local(file!.path!);
      if (!mounted) return;
      setState(() {
        _sdk = sdk;
        _modelName = file.name;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error);
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  Future<void> _ask() async {
    final sdk = _sdk;
    final prompt = _promptController.text.trim();
    if (sdk == null || prompt.isEmpty || _busy) {
      return;
    }

    setState(() {
      _busy = true;
      _error = null;
      _output = '';
    });

    try {
      await for (final token in sdk.askStream(prompt)) {
        if (!mounted) return;
        setState(() => _output += token);
      }
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error);
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Edge Intelligence',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xff0f766e)),
        useMaterial3: true,
      ),
      home: Scaffold(
        appBar: AppBar(title: const Text('Edge Intelligence')),
        body: SafeArea(
          child: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              const Text(
                'Legacy byte-level GGUF compatibility demo. Do not select '
                'Qwen2/Qwen2.5 models: this Dart binding is not '
                'tokenizer-aware.',
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: Text(
                      _modelName ?? 'No GGUF model selected',
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  const SizedBox(width: 12),
                  OutlinedButton.icon(
                    onPressed: _busy || widget.initializationError != null
                        ? null
                        : _selectModel,
                    icon: const Icon(Icons.folder_open),
                    label: const Text('Model'),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              TextField(
                controller: _promptController,
                enabled: !_busy,
                minLines: 3,
                maxLines: 6,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  labelText: 'Prompt',
                ),
              ),
              const SizedBox(height: 12),
              Align(
                alignment: Alignment.centerRight,
                child: FilledButton.icon(
                  onPressed: _sdk == null || _busy ? null : _ask,
                  icon: _busy
                      ? const SizedBox.square(
                          dimension: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.send),
                  label: const Text('Ask'),
                ),
              ),
              if (_error != null) ...[
                const SizedBox(height: 20),
                Text(
                  'Error: $_error',
                  style: TextStyle(color: Theme.of(context).colorScheme.error),
                ),
              ],
              if (_output.isNotEmpty) ...[
                const SizedBox(height: 20),
                SelectableText(_output),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
