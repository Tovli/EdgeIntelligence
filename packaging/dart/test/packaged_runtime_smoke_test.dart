import 'package:edge_intelligence/edge_intelligence.dart';
import 'package:test/test.dart';

void main() {
  test('loads the packaged native runtime', () async {
    await initEdgeIntelligence();
    addTearDown(disposeEdgeIntelligence);
  });
}
