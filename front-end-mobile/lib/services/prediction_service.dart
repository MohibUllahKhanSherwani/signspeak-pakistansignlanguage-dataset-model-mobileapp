import 'package:dio/dio.dart';

/// Model for prediction results from the FastAPI backend.
class PredictionResult {
  final String action;
  final double confidence;

  PredictionResult({
    required this.action,
    required this.confidence,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      action: json['action'] as String,
      confidence: (json['confidence'] as num).toDouble(),
    );
  }
}

/// Service for communicating with the FastAPI backend.
/// 
/// Handles:
/// - Sending 30-frame landmark buffers for prediction
/// - Health check for connection status
/// - Request throttling and error handling
class PredictionService {
  late final Dio _dio;
  final String baseUrl;
  
  // TODO: Replace with your actual backend IP address
  static const String defaultIp = '192.168.100.2'; // UPDATED TO MATCH YOUR BACKED IP
  static const int defaultPort = 8000;

  PredictionService({String? ipAddress, int? port})
      : baseUrl = 'http://${ipAddress ?? defaultIp}:${port ?? defaultPort}' {
    _dio = Dio(BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 5),
      receiveTimeout: const Duration(seconds: 5),
      sendTimeout: const Duration(seconds: 5),
      headers: {
        'Content-Type': 'application/json',
      },
    ));

    // Add logging interceptor for debugging
    _dio.interceptors.add(LogInterceptor(
      requestBody: false, // Set to true for detailed debugging
      responseBody: true,
      error: true,
    ));
  }

  /// Checks if the backend server is healthy and reachable.
  /// 
  /// Returns true if the server responds successfully, false otherwise.
  Future<bool> checkHealth() async {
    try {
      final response = await _dio.get('/health');
      return response.statusCode == 200;
    } catch (e) {
      print('Health check failed: $e');
      return false;
    }
  }

  /// Sends a 30-frame landmark buffer to the backend for prediction.
  /// 
  /// [landmarkBuffer]: List of 30 frames, each containing 126 float values
  /// 
  /// Returns a [PredictionResult] with the predicted action and confidence.
  /// Throws an exception if the request fails.
  Future<PredictionResult> predict(List<List<double>> landmarkBuffer) async {
    try {
      // Validate buffer
      if (landmarkBuffer.length != 60) {
        throw ArgumentError('Buffer must contain exactly 60 frames, got ${landmarkBuffer.length}');
      }

      for (int i = 0; i < landmarkBuffer.length; i++) {
        if (landmarkBuffer[i].length != 126) {
          throw ArgumentError('Frame $i must contain exactly 126 values, got ${landmarkBuffer[i].length}');
        }
      }

      // Send request
      // DEBUG: Print first frame data to verify
      if (landmarkBuffer.isNotEmpty && landmarkBuffer[0].isNotEmpty) {
        print('Sending buffer with ${landmarkBuffer.length} frames.');
        print('First frame (first 3 values - Left Wrist?): ${landmarkBuffer[0].sublist(0, 3)}');
        print('First frame (values 63-66 - Right Wrist?): ${landmarkBuffer[0].sublist(63, 66)}');
      }

      final response = await _dio.post(
        '/predict',
        data: {
          'landmarks': landmarkBuffer,
        },
      );

      if (response.statusCode == 200) {
        return PredictionResult.fromJson(response.data);
      } else {
        throw Exception('Prediction failed with status code: ${response.statusCode}');
      }
    } on DioException catch (e) {
      if (e.type == DioExceptionType.connectionTimeout) {
        throw Exception('Connection timeout - is the backend server running?');
      } else if (e.type == DioExceptionType.receiveTimeout) {
        throw Exception('Receive timeout - backend took too long to respond');
      } else if (e.type == DioExceptionType.connectionError) {
        throw Exception('Connection error - check backend IP address and network');
      } else {
        throw Exception('Request failed: ${e.message}');
      }
    } catch (e) {
      throw Exception('Unexpected error during prediction: $e');
    }
  }

  /// Gets the current backend URL being used.
  String getBackendUrl() => baseUrl;
}
