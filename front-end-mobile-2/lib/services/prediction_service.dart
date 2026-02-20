import 'dart:typed_data';
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config.dart';

enum InferenceModel { baseline, augmented }

extension InferenceModelX on InferenceModel {
  String get apiValue => name;

  String get label {
    switch (this) {
      case InferenceModel.baseline:
        return 'Baseline';
      case InferenceModel.augmented:
        return 'Augmented';
    }
  }
}

class PredictionResult {
  final String action;
  final double confidence;
  final Map<String, double> allProbabilities;
  final String modelUsed;
  final int processingTimeMs;
  final int framesProcessed;
  final int handsDetected;

  PredictionResult({
    required this.action,
    required this.confidence,
    required this.allProbabilities,
    required this.modelUsed,
    required this.processingTimeMs,
    required this.framesProcessed,
    required this.handsDetected,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      action: json['action'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      allProbabilities: (json['all_probabilities'] as Map<String, dynamic>).map(
        (k, v) => MapEntry(k, (v as num).toDouble()),
      ),
      modelUsed:
          (json['model_used'] as String?) ?? InferenceModel.baseline.apiValue,
      processingTimeMs: json['processing_time_ms'] as int,
      framesProcessed: json['frames_processed'] as int,
      handsDetected: json['hands_detected'] as int,
    );
  }
}

class PredictionService {
  /// Send a batch of JPEG frames to the backend for prediction.
  ///
  /// [frames] must be exactly 60 JPEG-encoded byte arrays.
  static Future<PredictionResult> predictFromFrames(
    List<Uint8List> frames, {
    required InferenceModel model,
  }) async {
    if (frames.length != AppConfig.sequenceLength) {
      throw Exception(
        'Expected ${AppConfig.sequenceLength} frames, got ${frames.length}',
      );
    }

    final uri = Uri.parse(
      '${AppConfig.serverUrl}/predict-frames?model=${model.apiValue}',
    );
    final request = http.MultipartRequest('POST', uri);

    for (int i = 0; i < frames.length; i++) {
      request.files.add(
        http.MultipartFile.fromBytes('frames', frames[i], filename: '$i.jpg'),
      );
    }

    final streamedResponse = await request.send().timeout(
      const Duration(seconds: 30),
    );
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode != 200) {
      throw Exception('Server error ${response.statusCode}: ${response.body}');
    }

    final json = jsonDecode(response.body) as Map<String, dynamic>;
    return PredictionResult.fromJson(json);
  }

  /// Check if the backend is reachable.
  static Future<bool> checkHealth() async {
    try {
      final response = await http
          .get(Uri.parse('${AppConfig.serverUrl}/health'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
