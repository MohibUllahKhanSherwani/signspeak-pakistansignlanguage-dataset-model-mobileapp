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
/// - Prediction smoothing (matching Python's sliding window logic)
/// - Sentence history of confirmed signs
/// - API communication throttling
class AppState extends ChangeNotifier {
  final LandmarkManager _landmarkManager = LandmarkManager();
  late final PredictionService _predictionService;
  
  bool _isConnected = false;
  String _currentAction = '';
  double _currentConfidence = 0.0;
  bool _isProcessing = false;
  String? _errorMessage;
  String _debugInfo = '';
  
  // --- Prediction Smoothing (matches Python realtime_inference_minimal.py) ---
  // Python keeps last 10 predictions, requires >8 agreements (9/10) + confidence > threshold
  // UPDATED: Relaxed to >5 agreements (6/10) + confidence > 0.7 for better responsiveness
  static const int _smoothingWindowSize = 10;
  static const int _requiredAgreements = 6;   // 6 out of 10 must agree
  static const double _confidenceThreshold = 0.7;
  static const int _maxSentenceLength = 20;
  
  final List<String> _recentPredictions = [];  // Sliding window of last N predictions
  String _confirmedAction = '';                 // The smoothed/stable prediction
  double _confirmedConfidence = 0.0;
  final List<String> _sentence = [];            // History of confirmed signs
  
  Timer? _throttleTimer;
  static const Duration _throttleDuration = Duration(milliseconds: 200);

  AppState({String? backendIp, int? backendPort}) {
    _predictionService = PredictionService(
      ipAddress: backendIp,
      port: backendPort,
    );
  }

  // Getters
  bool get isConnected => _isConnected;
  String get currentAction => _confirmedAction;        // Now returns SMOOTHED action
  double get currentConfidence => _confirmedConfidence; // Now returns SMOOTHED confidence
  String get rawAction => _currentAction;               // Raw (unsmoothed) for debugging
  double get rawConfidence => _currentConfidence;
  bool get isProcessing => _isProcessing;
  String? get errorMessage => _errorMessage;
  String get backendUrl => _predictionService.getBackendUrl();
  String get debugInfo => _debugInfo;
  int get currentFrameCount => _landmarkManager.currentFrameCount;
  bool get isBufferFull => _landmarkManager.isBufferFull;
  List<String> get sentence => List.unmodifiable(_sentence);
  int get smoothingProgress => _recentPredictions.isEmpty ? 0 : _countMostCommon();
  /// How many of the 60 buffered frames actually have hand data.
  int get nonZeroFrameCount => _landmarkManager.nonZeroFrameCount;
  static const int _minHandFrames = 20; // At least 20/60 frames must have hand data

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
    if (_landmarkManager.isBufferFull &&
        _throttleTimer == null &&
        !_isProcessing) {
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
      _isConnected = true;
      
      // --- Prediction Smoothing (mirrors Python logic) ---
      // Add to sliding window
      _recentPredictions.add(result.action);
      if (_recentPredictions.length > _smoothingWindowSize) {
        _recentPredictions.removeAt(0);
      }
      
      // Only confirm prediction if:
      // 1. Confidence exceeds threshold (0.8)
      // 2. At least 9 of last 10 predictions agree (>8)
      final agreementCount = _recentPredictions
          .where((p) => p == result.action)
          .length;
      
      if (result.confidence > _confidenceThreshold && 
          agreementCount >= _requiredAgreements) {
        _confirmedAction = result.action;
        _confirmedConfidence = result.confidence;
        
        // Add to sentence history (skip "nothing" and duplicates)
        if (result.action.toLowerCase() != 'nothing') {
          if (_sentence.isEmpty || _sentence.last != result.action) {
            _sentence.add(result.action);
            if (_sentence.length > _maxSentenceLength) {
              _sentence.removeRange(0, _sentence.length - _maxSentenceLength);
            }
          }
        }
      }
    } catch (e) {
      _errorMessage = e.toString();
      _isConnected = false;
      print('Prediction error: $e');
    } finally {
      _isProcessing = false;
      notifyListeners();
    }
  }
  
  /// Counts how many times the most common prediction appears in the window.
  int _countMostCommon() {
    if (_recentPredictions.isEmpty) return 0;
    final counts = <String, int>{};
    for (final p in _recentPredictions) {
      counts[p] = (counts[p] ?? 0) + 1;
    }
    return counts.values.reduce((a, b) => a > b ? a : b);
  }

  /// Clears the current prediction results and smoothing state.
  void clearPrediction() {
    _currentAction = '';
    _currentConfidence = 0.0;
    _confirmedAction = '';
    _confirmedConfidence = 0.0;
    _recentPredictions.clear();
    notifyListeners();
  }
  
  /// Clears the sentence history.
  void clearSentence() {
    _sentence.clear();
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
