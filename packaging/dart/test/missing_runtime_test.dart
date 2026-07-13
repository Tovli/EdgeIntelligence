import 'package:edge_intelligence/edge_intelligence.dart';
import 'package:test/test.dart';

void main() {
  test('reports a missing packaged runtime clearly', () async {
    await expectLater(initEdgeIntelligence(), throwsA(isA<UnsupportedError>()));
  });
}
