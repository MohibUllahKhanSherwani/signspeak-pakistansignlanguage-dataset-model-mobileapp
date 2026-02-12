import 'dart:async';
import 'package:flutter/foundation.dart';
import '../services/landmark_manager.dart';
import '../services/prediction_service.dart';

/// Application state for the Sign Language Recognition app.
/// 
/// Manages:
/// - Landmark buffer state
/// - Connection status to backend
/// - Current prediction results
/// - API communication throttling
class AppState extends ChangeNotifier {
  final LandmarkManager _landmarkManager = LandmarkManager();
  late final PredictionService _predictionService;
  
  bool _isConnected = false;
  String _currentAction = '';
  double _currentConfidence = 0.0;
  bool _isProcessing = false;
  String? _errorMessage;
  String _debugInfo = ''; // Add debug info field
  
  Timer? _throttleTimer;
  static const Duration _throttleDuration = Duration(milliseconds: 200); // 200ms throttle

  AppState({String? backendIp, int? backendPort}) {
    _predictionService = PredictionService(
      ipAddress: backendIp,
      port: backendPort,
    );
  }

  // Getters
  bool get isConnected => _isConnected;
  String get currentAction => _currentAction;
  double get currentConfidence => _currentConfidence;
  bool get isProcessing => _isProcessing;
  String? get errorMessage => _errorMessage;
  String get backendUrl => _predictionService.getBackendUrl();
  String get debugInfo => _debugInfo; // Add getter
  int get currentFrameCount => _landmarkManager.currentFrameCount;
  bool get isBufferFull => _landmarkManager.isBufferFull;

  /// Checks connection to the backend server.
  Future<void> checkConnection() async {
    try {
      _isConnected = await _predictionService.checkHealth();
      _errorMessage = null;
      notifyListeners();
    } catch (e) {
      _isConnected = false;
      _errorMessage = 'Connection check failed: $e';
      notifyListeners();
    }
  }

  /// Adds a new frame of landmarks to the buffer.
  /// 
  /// Automatically triggers prediction when buffer is full and throttle allows.
  void addLandmarkFrame({
    List<double>? leftHandLandmarks,
    List<double>? rightHandLandmarks,
  }) {
    _landmarkManager.addFrame(
      leftHandLandmarks: leftHandLandmarks,
      rightHandLandmarks: rightHandLandmarks,
    );

    // Trigger prediction if buffer is full and not currently throttled
    if (_landmarkManager.isBufferFull && _throttleTimer == null && !_isProcessing) {
      _schedulePrediction();
    }

    notifyListeners();
  }

  /// Schedules a prediction request with throttling.
  void _schedulePrediction() {
    _throttleTimer = Timer(_throttleDuration, () {
      _throttleTimer = null;
      _sendPrediction();
    });
  }

  /// Sends the current buffer to the backend for prediction.
  Future<void> _sendPrediction() async {
    if (!_landmarkManager.isBufferFull || _isProcessing) {
      return;
    }

    final buffer = _landmarkManager.getBuffer();
    if (buffer == null) {
      return;
    }

    _isProcessing = true;
    notifyListeners();

    try {
      final result = await _predictionService.predict(buffer);
      _currentAction = result.action;
      _currentConfidence = result.confidence;
      _errorMessage = null;
      _isConnected = true; // Update connection status on successful request
    } catch (e) {
      _errorMessage = e.toString();
      _isConnected = false;
      print('Prediction error: $e');
    } finally {
      _isProcessing = false;
      notifyListeners();
    }
  }

  /// Clears the current prediction results.
  void clearPrediction() {
    _currentAction = '';
    _currentConfidence = 0.0;
    notifyListeners();
  }

  /// Resets the landmark buffer.
  void resetBuffer() {
    _landmarkManager.clear();
    notifyListeners();
  }

  /// Gets buffer statistics for debugging.
  Map<String, dynamic> getBufferStats() {
    return _landmarkManager.getStats();
  }

  void updateDebugInfo(String info) {
    _debugInfo = info;
    notifyListeners();
  }

  @override
  void dispose() {
    _throttleTimer?.cancel();
    super.dispose();
  }
}
