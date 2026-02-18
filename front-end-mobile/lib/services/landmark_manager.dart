import 'dart:collection';

/// Manages a sliding window buffer of hand landmarks for sign language recognition.
/// 
/// The buffer maintains exactly 60 frames of landmark data. Each frame contains
/// 126 float values representing:
/// - Positions 0-62: Left Hand landmarks (21 points × 3 coordinates: x, y, z)
/// - Positions 63-125: Right Hand landmarks (21 points × 3 coordinates: x, y, z)
/// 
/// If a hand is not detected in a frame, the corresponding 63 values are set to 0.0.
class LandmarkManager {
  static const int _bufferSize = 60;
  static const int _landmarksPerFrame = 126;
  static const int _landmarksPerHand = 63; // 21 points × 3 coordinates

  final Queue<List<double>> _buffer = Queue<List<double>>();

  /// Adds a new frame of landmarks to the buffer.
  /// 
  /// [leftHandLandmarks]: List of 3D coordinates for left hand (21 points × 3 = 63 values)
  /// [rightHandLandmarks]: List of 3D coordinates for right hand (21 points × 3 = 63 values)
  /// 
  /// If a hand is not detected, pass null and zeros will be used for that hand.
  void addFrame({
    List<double>? leftHandLandmarks,
    List<double>? rightHandLandmarks,
  }) {
    // Prepare left hand data (63 values)
    final leftHand = leftHandLandmarks != null && leftHandLandmarks.length == _landmarksPerHand
        ? leftHandLandmarks
        : List<double>.filled(_landmarksPerHand, 0.0);

    // Prepare right hand data (63 values)
    final rightHand = rightHandLandmarks != null && rightHandLandmarks.length == _landmarksPerHand
        ? rightHandLandmarks
        : List<double>.filled(_landmarksPerHand, 0.0);

    // Combine into single frame (126 values)
    final frame = [...leftHand, ...rightHand];

    // Add to buffer
    _buffer.add(frame);

    // Maintain buffer size at 60 frames
    if (_buffer.length > _bufferSize) {
      _buffer.removeFirst();
    }
  }

  /// Returns true if the buffer is full (contains 60 frames).
  bool get isBufferFull => _buffer.length == _bufferSize;

  /// Returns the current number of frames in the buffer.
  int get currentFrameCount => _buffer.length;

  /// Gets the complete 60-frame buffer as a list of lists.
  /// 
  /// Returns null if the buffer is not yet full.
  /// Format: [[frame1 (126 values)], [frame2 (126 values)], ..., [frame60 (126 values)]]
  List<List<double>>? getBuffer() {
    if (!isBufferFull) {
      return null;
    }
    return _buffer.toList();
  }

  /// Clears the buffer.
  void clear() {
    _buffer.clear();
  }

  /// Returns buffer statistics for debugging.
  Map<String, dynamic> getStats() {
    return {
      'bufferSize': _buffer.length,
      'isFull': isBufferFull,
      'totalFrames': _bufferSize,
      'landmarksPerFrame': _landmarksPerFrame,
    };
  }
}
