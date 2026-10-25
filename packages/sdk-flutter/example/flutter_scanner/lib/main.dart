import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:khmerocr/khmerocr.dart';

// Defaults can be passed at build time instead of typed in the app:
//   flutter run --dart-define=KHMEROCR_URL=http://192.168.1.10:8000 \
//               --dart-define=KHMEROCR_API_KEY=...
// 10.0.2.2 is the host machine as seen from the Android emulator.
const _defaultUrl = String.fromEnvironment('KHMEROCR_URL',
    defaultValue: 'http://10.0.2.2:8000');
const _defaultApiKey = String.fromEnvironment('KHMEROCR_API_KEY');

void main() => runApp(const ScannerApp());

class ScannerApp extends StatelessWidget {
  const ScannerApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'KhmerOCR scanner',
        theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
        home: const ScannerPage(),
      );
}

enum CardType { nationalId, studentId }

const _fieldLabels = {
  'id_number': 'ID number',
  'full_name_km': 'Name (Khmer)',
  'full_name_latin': 'Name (Latin)',
  'date_of_birth': 'Date of birth',
  'gender': 'Gender',
  'department': 'Department',
  'degree': 'Degree',
  'address_km': 'Address',
  'student_id': 'Student ID',
  'issue_date': 'Issue date',
};

class ScannerPage extends StatefulWidget {
  const ScannerPage({super.key});

  @override
  State<ScannerPage> createState() => _ScannerPageState();
}

class _ScannerPageState extends State<ScannerPage> {
  final _url = TextEditingController(text: _defaultUrl);
  final _apiKey = TextEditingController(text: _defaultApiKey);
  final _picker = ImagePicker();
  CardType _cardType = CardType.nationalId;
  bool _busy = false;
  DocumentResult? _result;
  String? _error;

  @override
  void dispose() {
    _url.dispose();
    _apiKey.dispose();
    super.dispose();
  }

  Future<void> _scan(ImageSource source) async {
    // Downscaled JPEG keeps uploads well under the 10 MiB server limit while
    // leaving enough resolution for the card text.
    final photo = await _picker.pickImage(
        source: source, maxWidth: 2000, imageQuality: 90);
    if (photo == null) return;
    setState(() {
      _busy = true;
      _result = null;
      _error = null;
    });
    final client = KhmerOcrClient(
      baseUrl: Uri.parse(_url.text.trim()),
      apiKey: _apiKey.text.trim().isEmpty ? null : _apiKey.text.trim(),
    );
    try {
      // The image is only held in memory for this request; it is not saved,
      // logged or cached by the app.
      final image = await photo.readAsBytes();
      final result = _cardType == CardType.nationalId
          ? await client.extractNationalId(image)
          : await client.extractStudentId(image);
      setState(() => _result = result);
    } on KhmerOcrException catch (error) {
      setState(() => _error = _describe(error));
    } on FormatException {
      setState(() => _error = 'The server URL is not valid.');
    } finally {
      client.close();
      setState(() => _busy = false);
    }
  }

  String _describe(KhmerOcrException error) {
    final message = switch (error.code) {
      KhmerOcrErrorCode.imageQualityTooLow =>
        'The photo is not clear enough. Retake it in even light, without glare, '
            'with the whole card in the frame.',
      KhmerOcrErrorCode.unauthorized => 'The API key is missing or wrong.',
      KhmerOcrErrorCode.networkError =>
        'Could not reach the server. Check the URL and that the phone and '
            'server are on the same network.',
      KhmerOcrErrorCode.timeout =>
        'The server took too long to answer. Try again.',
      KhmerOcrErrorCode.modelNotAvailable =>
        'The server has no OCR model loaded. Ask its administrator.',
      KhmerOcrErrorCode.payloadTooLarge => 'The photo is too large.',
      KhmerOcrErrorCode.unsupportedMediaType ||
      KhmerOcrErrorCode.invalidImage =>
        'This file is not a supported image (JPEG, PNG or WebP).',
      _ => 'Something went wrong (${error.code}).',
    };
    final id =
        error.requestId == null ? '' : '\nRequest ID: ${error.requestId}';
    return '$message$id';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('KhmerOCR scanner')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(
            controller: _url,
            decoration: const InputDecoration(labelText: 'Server URL'),
            keyboardType: TextInputType.url,
            autocorrect: false,
          ),
          TextField(
            controller: _apiKey,
            decoration: const InputDecoration(labelText: 'API key'),
            obscureText: true,
            autocorrect: false,
          ),
          const SizedBox(height: 16),
          SegmentedButton<CardType>(
            segments: const [
              ButtonSegment(
                  value: CardType.nationalId, label: Text('National ID')),
              ButtonSegment(
                  value: CardType.studentId, label: Text('Student ID')),
            ],
            selected: {_cardType},
            onSelectionChanged: (value) =>
                setState(() => _cardType = value.first),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: _busy ? null : () => _scan(ImageSource.camera),
                  icon: const Icon(Icons.photo_camera),
                  label: const Text('Take photo'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _busy ? null : () => _scan(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),
          if (_busy) const Center(child: CircularProgressIndicator()),
          if (_error != null) _ErrorCard(message: _error!),
          if (_result != null) _ResultView(result: _result!),
        ],
      ),
    );
  }
}

class _ErrorCard extends StatelessWidget {
  const _ErrorCard({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) => Card(
        color: Theme.of(context).colorScheme.errorContainer,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Text(message),
        ),
      );
}

class _ResultView extends StatelessWidget {
  const _ResultView({required this.result});

  final DocumentResult result;

  @override
  Widget build(BuildContext context) {
    final needsReview = result.fieldsNeedingReview.toSet();
    final known = result.documentType == 'student_id_card'
        ? StudentIdField.all
        : NationalIdField.all;
    final names = [
      ...known.where(result.fields.containsKey),
      ...result.fields.keys.where((name) => !known.contains(name)),
    ];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _StatusBanner(result: result),
        const SizedBox(height: 8),
        for (final name in names)
          _FieldTile(
            label: _fieldLabels[name] ?? name,
            field: result.fields[name]!,
            check: needsReview.contains(name),
          ),
      ],
    );
  }
}

class _StatusBanner extends StatelessWidget {
  const _StatusBanner({required this.result});

  final DocumentResult result;

  @override
  Widget build(BuildContext context) {
    final (color, icon, text) = switch (result.status) {
      ResultStatus.failed => (
          Colors.red.shade100,
          Icons.error,
          'Nothing could be read. Retake the photo.'
        ),
      _ when result.reviewRequired => (
          Colors.amber.shade100,
          Icons.fact_check,
          'Check every field against the card before using it.'
        ),
      _ => (Colors.green.shade100, Icons.check_circle, 'Read successfully.'),
    };
    return Card(
      color: color,
      child: ListTile(leading: Icon(icon), title: Text(text)),
    );
  }
}

class _FieldTile extends StatelessWidget {
  const _FieldTile(
      {required this.label, required this.field, required this.check});

  final String label;
  final ExtractedField field;
  final bool check;

  @override
  Widget build(BuildContext context) {
    final confidence = field.confidence == null
        ? ''
        : ' · ${(field.confidence! * 100).round()}% confidence';
    return Card(
      color: check ? Colors.orange.shade50 : null,
      shape: check
          ? RoundedRectangleBorder(
              side: BorderSide(color: Colors.orange.shade700, width: 1.5),
              borderRadius: BorderRadius.circular(12),
            )
          : null,
      child: ListTile(
        title: Text(field.value ?? 'Not read',
            style: TextStyle(
                fontStyle: field.value == null ? FontStyle.italic : null)),
        subtitle: Text('$label$confidence'),
        trailing: check
            ? Tooltip(
                message: 'Check this field',
                child: Icon(Icons.warning_amber, color: Colors.orange.shade800),
              )
            : null,
      ),
    );
  }
}
