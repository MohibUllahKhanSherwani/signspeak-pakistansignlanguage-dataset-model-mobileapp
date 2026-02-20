/// Server URL and app configuration
class AppConfig {
  // Change this to your laptop's local IP address
  static const String serverUrl = 'http://192.168.100.2:8000';
  
  // Frame capture settings
  static const int sequenceLength = 60;        // Must match training SEQUENCE_LENGTH
  static const int targetFps = 30;             // Match training webcam FPS
  static const int recordDurationMs = 2000;    // 2 seconds
  static const int frameIntervalMs = 1000 ~/ targetFps; // ~33ms between frames
  
  // JPEG compression settings 
  static const int jpegQuality = 70;           // Good quality, reasonable size
  static const int frameWidth = 320;           // Resize width for upload
  static const int frameHeight = 240;          // Resize height for upload
  
  // Prediction display
  static const double confidenceThreshold = 0.7;
}
