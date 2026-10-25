import 'dart:typed_data';

/// Image formats the server accepts.
enum ImageType {
  jpeg('image/jpeg', 'jpg'),
  png('image/png', 'png'),
  webp('image/webp', 'webp');

  const ImageType(this.mimeType, this.extension);

  final String mimeType;
  final String extension;

  static ImageType? fromMimeType(String mimeType) {
    final normalized = mimeType.split(';').first.trim().toLowerCase();
    for (final type in values) {
      if (type.mimeType == normalized) return type;
    }
    return null;
  }

  /// Identifies the format from the file's magic bytes.
  static ImageType? sniff(Uint8List bytes) {
    bool startsWith(List<int> prefix, [int offset = 0]) {
      if (bytes.length < offset + prefix.length) return false;
      for (var i = 0; i < prefix.length; i++) {
        if (bytes[offset + i] != prefix[i]) return false;
      }
      return true;
    }

    if (startsWith(const [0xFF, 0xD8, 0xFF])) return jpeg;
    if (startsWith(const [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])) {
      return png;
    }
    if (startsWith('RIFF'.codeUnits) && startsWith('WEBP'.codeUnits, 8)) {
      return webp;
    }
    return null;
  }
}
