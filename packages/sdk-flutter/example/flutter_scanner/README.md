# KhmerOCR scanner (Flutter example)

Takes a photo of a National ID or student ID card, sends it to a KhmerOCR
server, and shows each field with its confidence. Fields to check are
outlined in orange, and a banner says when the result needs review.

The repository holds only `lib/`, `test/` and `pubspec.yaml`. Generate the
platform folders once:

```bash
cd packages/sdk-flutter/example/flutter_scanner
flutter create --platforms=android,ios .
flutter pub get
```

## Run a development server

Start the server with an API key, listening on your network:

```bash
export KHMEROCR_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
khmerocr serve --host 0.0.0.0 --port 8000 --model khmerocr-id-v1
```

| Device | Server URL |
| --- | --- |
| Android emulator | `http://10.0.2.2:8000` (the host machine) |
| iOS simulator | `http://127.0.0.1:8000` |
| Real phone | `http://<computer's LAN IP>:8000`, same Wi-Fi network |

Type the URL and key in the app, or pass them at build time:

```bash
flutter run --dart-define=KHMEROCR_URL=http://192.168.1.10:8000 \
            --dart-define=KHMEROCR_API_KEY=...
```

## Platform setup

**Android** (`android/app/src/main/AndroidManifest.xml`): add
`<uses-permission android:name="android.permission.INTERNET"/>` above
`<application>`. Plain `http://` is blocked by default; for local
development only, add `android:usesCleartextTraffic="true"` to
`<application>` (or a network security config allowing just your dev host).
Remove it for release builds and use HTTPS.

**iOS** (`ios/Runner/Info.plist`): add `NSCameraUsageDescription` and
`NSPhotoLibraryUsageDescription` with a sentence explaining that the app
photographs ID cards. For local `http://` development only, add an
`NSAppTransportSecurity` exception (`NSAllowsLocalNetworking` = true). Use
HTTPS in release builds.

The app keeps the photo in memory for the request only. It does not save,
log or cache it. Don't ship a build with a real API key in it; see the
security notes in the [package README](../../README.md#security).
