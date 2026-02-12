# SignSpeak - Real-Time Sign Language Recognition App

## Quick Start Guide

### Prerequisites Checklist

Before running the app, ensure you have:

- ✅ **Flutter SDK** installed and in your PATH
  - Verify: Run `flutter doctor` in terminal
  - If not installed: Visit https://docs.flutter.dev/get-started/install
  
- ✅ **Android Studio** or **Xcode** (for iOS)
  - For Android development
  - For iOS development (macOS only)

- ✅ **FastAPI Backend** running
  - Your backend should be accessible at the IP address configured in `lib/main.dart`
  - Default: `http://192.168.1.100:8000`

### Setup Instructions

1. **Install Dependencies**
   ```bash
   flutter pub get
   ```

2. **Configure Backend IP Address**
   
   Open `lib/main.dart` and update line 30:
   ```dart
   backendIp: 'YOUR_BACKEND_IP_HERE', // Change this to your backend IP
   ```

3. **Run the App**

   For Android:
   ```bash
   flutter run
   ```

   For iOS (macOS only):
   ```bash
   flutter run -d ios
   ```

### Architecture Overview

```
Sign Language Recognition App
│
├── Services Layer
│   ├── landmark_manager.dart      # Manages 30-frame sliding window buffer
│   └── prediction_service.dart    # Handles FastAPI backend communication
│
├── Providers
│   └── app_state.dart             # Application state management (Provider)
│
├── Widgets
│   ├── camera_view.dart           # Camera preview + prediction overlay
│   └── connection_indicator.dart  # Backend connection status
│
└── main.dart                      # App entry point + camera + MLKit integration
```

### How It Works

1. **Camera Feed**: Captures real-time video at ~30 FPS using the device's front camera

2. **Hand Landmark Extraction**: 
   - Uses Google MLKit Pose Detection on-device
   - Extracts hand landmarks (wrist, thumb, index, pinky)
   - Formats into 126 float values per frame:
     - Positions 0-62: Left Hand (21 points × 3 coordinates: x, y, z)
     - Positions 63-125: Right Hand (21 points × 3 coordinates: x, y, z)
   - Missing hands are padded with zeros

3. **Sliding Window Buffer**:
   - Maintains last 30 frames of landmarks
   - Automatically managed by `LandmarkManager`

4. **API Communication**:
   - Sends buffer to FastAPI backend every 200ms (throttled)
   - Endpoint: `POST /predict`
   - Payload: `{"landmarks": [[...30 frames of 126 values each...]]}`

5. **Prediction Display**:
   - Shows detected sign language action
   - Displays confidence percentage with color-coded progress bar
   - Green (≥80%), Orange (≥50%), Red (<50%)

### Configuration

#### Backend Connection

Edit `lib/services/prediction_service.dart`:
```dart
static const String defaultIp = 'YOUR_BACKEND_IP';  // Line 30
static const int defaultPort = 8000;                 // Line 31
```

Or pass it when creating `AppState` in `lib/main.dart`:
```dart
create: (_) => AppState(
  backendIp: 'YOUR_BACKEND_IP',
  backendPort: 8000,
),
```

#### Throttle Timing

Edit `lib/providers/app_state.dart`:
```dart
static const Duration _throttleDuration = Duration(milliseconds: 200);  // Line 22
```

Adjust between 150-300ms based on your needs:
- Lower (150ms): More responsive, higher network usage
- Higher (300ms): Less network usage, slightly delayed

### Troubleshooting

#### Camera Not Working
- **Android**: Check camera permissions in device settings
- **iOS**: Ensure `NSCameraUsageDescription` is in `ios/Runner/Info.plist`

#### Backend Connection Issues
- Verify backend is running: `curl http://YOUR_IP:8000/health`
- Check if app and backend are on the same network
- Disable firewall temporarily to test
- Update IP address in `main.dart`

#### MLKit Issues
- Ensure you have an internet connection for first-time MLKit model download
- Check device has sufficient storage
- Restart the app

#### Build Errors
```bash
# Clean and rebuild
flutter clean
flutter pub get
flutter run
```

### Testing

The app includes connection status indicator at the top-right:
- 🟢 **Green "Connected"**: Backend is reachable
- 🔴 **Red "Disconnected"**: Cannot reach backend

Buffer status shows at top-left (debug info):
- Shows current frame count (e.g., "Buffer: 28/30")

### Important Notes

> **⚠️ Hand Landmark Limitation**: The current implementation uses MLKit Pose Detection which provides basic hand landmarks (wrist, thumb, index, pinky). For full 21-point hand tracking, consider integrating MediaPipe Hands library for better accuracy.

> **📱 Device Requirements**: This app requires a physical device with a camera. It will not work properly in emulators.

> **🔋 Battery Usage**: Real-time camera processing and ML inference consume significant battery. This is expected for this type of application.

### Project Structure

```
lib/
├── main.dart                    # App entry point + camera integration
├── providers/
│   └── app_state.dart          # State management
├── services/
│   ├── landmark_manager.dart   # Buffer management
│   └── prediction_service.dart # API client
└── widgets/
    ├── camera_view.dart        # Camera UI
    └── connection_indicator.dart # Status indicator
```

### Dependencies

See `pubspec.yaml` for full list:
- `camera`: Camera access
- `google_mlkit_pose_detection`: Hand landmark detection
- `dio`: HTTP client
- `provider`: State management
- `permission_handler`: Runtime permissions

### Next Steps

1. **Install Flutter** if you haven't already
2. **Run `flutter pub get`** to install dependencies
3. **Update backend IP** in `lib/main.dart`
4. **Start your FastAPI backend**
5. **Run the app** with `flutter run`

### Support

For issues or questions about:
- **Flutter Setup**: https://docs.flutter.dev/
- **MLKit**: https://pub.dev/packages/google_mlkit_pose_detection
- **Camera Plugin**: https://pub.dev/packages/camera

---

**Built with Flutter 💙 | SignSpeak - Pakistan Sign Language Recognition**
