import 'package:edge_intelligence_example/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('shows the mobile SDK controls', (tester) async {
    await tester.pumpWidget(
      const EdgeIntelligenceExample(initializationError: 'runtime unavailable'),
    );

    expect(find.text('Edge Intelligence'), findsOneWidget);
    expect(find.text('Model'), findsOneWidget);
    expect(find.text('Prompt'), findsOneWidget);
    expect(find.text('Ask'), findsOneWidget);
  });
}
