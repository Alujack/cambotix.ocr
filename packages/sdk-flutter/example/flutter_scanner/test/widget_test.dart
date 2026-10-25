import 'package:flutter/material.dart';
import 'package:flutter_scanner/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('shows the server settings and scan buttons', (tester) async {
    await tester.pumpWidget(const ScannerApp());
    expect(find.text('Server URL'), findsOneWidget);
    expect(find.text('Take photo'), findsOneWidget);
    expect(find.text('Gallery'), findsOneWidget);

    await tester.tap(find.text('Student ID'));
    await tester.pump();
    final button = tester.widget<SegmentedButton<CardType>>(
        find.byType(SegmentedButton<CardType>));
    expect(button.selected, {CardType.studentId});
  });
}
