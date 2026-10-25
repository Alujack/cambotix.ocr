/// Response models mirroring the JSON Schemas in the repository's `schemas/`.
///
/// Every model keeps the decoded JSON in [json] so fields added by newer
/// servers stay reachable.
library;

/// Field names of a Cambodian National ID result.
abstract final class NationalIdField {
  static const idNumber = 'id_number';
  static const fullNameKm = 'full_name_km';
  static const fullNameLatin = 'full_name_latin';
  static const dateOfBirth = 'date_of_birth';
  static const gender = 'gender';

  static const all = [idNumber, fullNameKm, fullNameLatin, dateOfBirth, gender];
}

/// Field names of a university student ID card result.
abstract final class StudentIdField {
  static const fullNameKm = 'full_name_km';
  static const fullNameLatin = 'full_name_latin';
  static const gender = 'gender';
  static const dateOfBirth = 'date_of_birth';
  static const department = 'department';
  static const degree = 'degree';
  static const addressKm = 'address_km';
  static const studentId = 'student_id';
  static const issueDate = 'issue_date';

  static const all = [
    fullNameKm,
    fullNameLatin,
    gender,
    dateOfBirth,
    department,
    degree,
    addressKm,
    studentId,
    issueDate,
  ];
}

/// Overall outcome of a request.
enum ResultStatus {
  /// No review needed.
  success,

  /// A person must confirm the result before it is used.
  reviewRequired,

  /// Nothing usable could be read.
  failed,

  /// A status this client version does not know.
  unknown;

  static ResultStatus parse(Object? value) => switch (value) {
        'success' => success,
        'review_required' => reviewRequired,
        'failed' => failed,
        _ => unknown,
      };
}

class BoundingBox {
  const BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    this.json = const {},
  });

  factory BoundingBox.fromJson(Map<String, dynamic> json) => BoundingBox(
        x: _int(json['x']),
        y: _int(json['y']),
        width: _int(json['width']),
        height: _int(json['height']),
        json: json,
      );

  final int x;
  final int y;
  final int width;
  final int height;
  final Map<String, dynamic> json;
}

/// Text read in one region of the image.
class TextRegion {
  const TextRegion({
    required this.text,
    required this.confidence,
    required this.bbox,
    this.label,
    this.json = const {},
  });

  factory TextRegion.fromJson(Map<String, dynamic> json) => TextRegion(
        text: json['text'] as String? ?? '',
        confidence: _double(json['confidence']),
        bbox: BoundingBox.fromJson(_map(json['bbox']) ?? const {}),
        label: json['label'] as String?,
        json: json,
      );

  final String text;
  final double? confidence;
  final BoundingBox bbox;

  /// The template field this region belongs to, if any.
  final String? label;
  final Map<String, dynamic> json;
}

/// One extracted field.
class ExtractedField {
  const ExtractedField({
    required this.value,
    required this.confidence,
    required this.source,
    this.valid,
    this.json = const {},
  });

  factory ExtractedField.fromJson(Map<String, dynamic> json) => ExtractedField(
        value: json['value'] as String?,
        confidence: _double(json['confidence']),
        source: json['source'] as String? ?? '',
        valid: json['valid'] as bool?,
        json: json,
      );

  /// Normalized value (ISO-8601 dates, uppercase Latin names, `M`/`F`), or
  /// null when nothing usable was read.
  final String? value;

  /// Recognition confidence for the text read in this field's region. It
  /// does not measure whether the region was mapped to the right field.
  final double? confidence;
  final String source;

  /// Whether the value passed format validation for its field type.
  final bool? valid;
  final Map<String, dynamic> json;
}

/// A warning split into its code and, for field warnings, the field name.
class ResultWarning {
  const ResultWarning(this.code, [this.field]);

  /// Parses `code` or `code:field`.
  factory ResultWarning.parse(String warning) {
    final index = warning.indexOf(':');
    if (index < 0) return ResultWarning(warning);
    return ResultWarning(
        warning.substring(0, index), warning.substring(index + 1));
  }

  final String code;
  final String? field;

  @override
  String toString() => field == null ? code : '$code:$field';
}

/// Field extraction result for a card document.
class DocumentResult {
  const DocumentResult({
    required this.schemaVersion,
    required this.documentType,
    required this.status,
    required this.fields,
    required this.warnings,
    required this.reviewRequired,
    this.metadata,
    this.textRegions,
    this.json = const {},
  });

  factory DocumentResult.fromJson(Map<String, dynamic> json) => DocumentResult(
        schemaVersion: json['schema_version'] as String? ?? '',
        documentType: json['document_type'] as String? ?? '',
        status: ResultStatus.parse(json['status']),
        fields: {
          for (final entry in (_map(json['fields']) ?? const {}).entries)
            entry.key: ExtractedField.fromJson(_map(entry.value) ?? const {}),
        },
        warnings: _strings(json['warnings']),
        reviewRequired: json['review_required'] as bool? ?? true,
        metadata: _map(json['metadata']),
        textRegions: json['text_regions'] == null
            ? null
            : _maps(json['text_regions']).map(TextRegion.fromJson).toList(),
        json: json,
      );

  final String schemaVersion;

  /// `khmer_national_id` or `student_id_card`.
  final String documentType;
  final ResultStatus status;
  final Map<String, ExtractedField> fields;
  final List<String> warnings;

  /// When true, a person must confirm the result before it is used.
  final bool reviewRequired;
  final Map<String, dynamic>? metadata;

  /// Raw text of every template region; only present when requested.
  final List<TextRegion>? textRegions;
  final Map<String, dynamic> json;

  /// The field named [name], e.g. [NationalIdField.idNumber].
  ExtractedField? field(String name) => fields[name];

  List<ResultWarning> get parsedWarnings =>
      warnings.map(ResultWarning.parse).toList();

  /// Fields to highlight for a person to check: those named in a warning,
  /// plus those that failed validation or were not read. Use
  /// [reviewRequired] to decide whether the whole result needs review.
  List<String> get fieldsNeedingReview {
    final names = <String>{
      for (final warning in parsedWarnings)
        if (warning.field != null) warning.field!,
      for (final entry in fields.entries)
        if (entry.value.valid == false || entry.value.value == null) entry.key,
    };
    return names.toList();
  }
}

/// General text recognition result.
class OcrResult {
  const OcrResult({
    required this.schemaVersion,
    required this.documentType,
    required this.status,
    required this.text,
    required this.regions,
    required this.warnings,
    required this.reviewRequired,
    this.metadata,
    this.json = const {},
  });

  factory OcrResult.fromJson(Map<String, dynamic> json) => OcrResult(
        schemaVersion: json['schema_version'] as String? ?? '',
        documentType: json['document_type'] as String? ?? '',
        status: ResultStatus.parse(json['status']),
        text: json['text'] as String? ?? '',
        regions: _maps(json['regions']).map(TextRegion.fromJson).toList(),
        warnings: _strings(json['warnings']),
        reviewRequired: json['review_required'] as bool? ?? true,
        metadata: _map(json['metadata']),
        json: json,
      );

  final String schemaVersion;
  final String documentType;
  final ResultStatus status;
  final String text;
  final List<TextRegion> regions;
  final List<String> warnings;
  final bool reviewRequired;
  final Map<String, dynamic>? metadata;
  final Map<String, dynamic> json;
}

/// Whether an image is usable for extraction, and why not.
class QualityReport {
  const QualityReport({
    required this.schemaVersion,
    required this.usable,
    required this.width,
    required this.height,
    required this.blurScore,
    required this.glareRatio,
    required this.brightness,
    required this.contrast,
    required this.issues,
    required this.blockingIssues,
    this.json = const {},
  });

  factory QualityReport.fromJson(Map<String, dynamic> json) => QualityReport(
        schemaVersion: json['schema_version'] as String? ?? '',
        usable: json['usable'] as bool? ?? false,
        width: _int(json['width']),
        height: _int(json['height']),
        blurScore: _double(json['blur_score']) ?? 0,
        glareRatio: _double(json['glare_ratio']) ?? 0,
        brightness: _double(json['brightness']) ?? 0,
        contrast: _double(json['contrast']) ?? 0,
        issues: _strings(json['issues']),
        blockingIssues: _strings(json['blocking_issues']),
        json: json,
      );

  final String schemaVersion;
  final bool usable;
  final int width;
  final int height;
  final double blurScore;
  final double glareRatio;
  final double brightness;
  final double contrast;
  final List<String> issues;

  /// Issues that make extraction fail with `IMAGE_QUALITY_TOO_LOW`.
  final List<String> blockingIssues;
  final Map<String, dynamic> json;
}

/// `GET /v1/health`.
class HealthStatus {
  const HealthStatus({
    required this.status,
    required this.modelLoaded,
    required this.apiVersion,
    required this.sdkVersion,
    this.modelError,
    this.json = const {},
  });

  factory HealthStatus.fromJson(Map<String, dynamic> json) => HealthStatus(
        status: json['status'] as String? ?? '',
        modelLoaded: json['model_loaded'] as bool? ?? false,
        apiVersion: json['api_version'] as String? ?? '',
        sdkVersion: json['sdk_version'] as String? ?? '',
        modelError: json['model_error'] as String?,
        json: json,
      );

  final String status;
  final bool modelLoaded;
  final String apiVersion;
  final String sdkVersion;

  /// Error code explaining why no model is loaded.
  final String? modelError;
  final Map<String, dynamic> json;
}

/// Public description of an installed model bundle.
class ModelSummary {
  const ModelSummary({
    required this.modelId,
    required this.modelVersion,
    this.task,
    this.runtime,
    this.recognitionModel,
    this.schemaVersion,
    this.license,
    this.licenseApproved,
    this.recognizerBackend,
    this.weightsSha256,
    this.json = const {},
  });

  factory ModelSummary.fromJson(Map<String, dynamic> json) => ModelSummary(
        modelId: json['model_id'] as String? ?? '',
        modelVersion: json['model_version'] as String? ?? '',
        task: json['task'] as String?,
        runtime: json['runtime'] as String?,
        recognitionModel: json['recognition_model'] as String?,
        schemaVersion: json['schema_version'] as String?,
        license: json['license'] as String?,
        licenseApproved: json['license_approved'] as bool?,
        recognizerBackend: json['recognizer_backend'] as String?,
        weightsSha256: json['weights_sha256'] as String?,
        json: json,
      );

  final String modelId;
  final String modelVersion;
  final String? task;
  final String? runtime;
  final String? recognitionModel;
  final String? schemaVersion;
  final String? license;
  final bool? licenseApproved;
  final String? recognizerBackend;
  final String? weightsSha256;
  final Map<String, dynamic> json;
}

/// The model serving requests and the installed bundles (`GET /v1/models`).
class ModelsInfo {
  const ModelsInfo(
      {required this.active, required this.installed, this.json = const {}});

  factory ModelsInfo.fromJson(Map<String, dynamic> json) {
    final active = _map(json['active']);
    return ModelsInfo(
      active: active == null ? null : ActiveModel.fromJson(active),
      installed: _maps(json['installed']).map(ModelSummary.fromJson).toList(),
      json: json,
    );
  }

  /// Null when the server has no model loaded.
  final ActiveModel? active;
  final List<ModelSummary> installed;
  final Map<String, dynamic> json;
}

/// The engine currently serving requests.
class ActiveModel {
  const ActiveModel({
    required this.sdkVersion,
    required this.model,
    required this.recognizer,
    required this.templates,
    this.json = const {},
  });

  factory ActiveModel.fromJson(Map<String, dynamic> json) {
    final model = _map(json['model']);
    return ActiveModel(
      sdkVersion: json['sdk_version'] as String? ?? '',
      model: model == null ? null : ModelSummary.fromJson(model),
      recognizer: _map(json['recognizer']) ?? const {},
      templates: _map(json['templates']) ?? const {},
      json: json,
    );
  }

  final String sdkVersion;
  final ModelSummary? model;

  /// Backend description; its keys depend on the backend.
  final Map<String, dynamic> recognizer;

  /// Supported document types and their layout templates.
  final Map<String, dynamic> templates;
  final Map<String, dynamic> json;
}

int _int(Object? value) => value is num ? value.toInt() : 0;

double? _double(Object? value) => value is num ? value.toDouble() : null;

Map<String, dynamic>? _map(Object? value) =>
    value is Map ? value.cast<String, dynamic>() : null;

List<Map<String, dynamic>> _maps(Object? value) => value is List
    ? [
        for (final item in value)
          if (item is Map) item.cast<String, dynamic>()
      ]
    : const [];

List<String> _strings(Object? value) => value is List
    ? [
        for (final item in value)
          if (item is String) item
      ]
    : const [];
