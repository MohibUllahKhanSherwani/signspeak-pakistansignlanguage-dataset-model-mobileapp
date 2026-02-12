import 'package:flutter_test/flutter_test.dart';
import 'package:sign_speak_mob_app/services/prediction_service.dart';

void main() {
  group('PredictionService Tests', () {
    late PredictionService service;

    setUp(() {
      service = PredictionService(
        ipAddress: '192.168.100.2',
        port: 8000,
      );
    });

    test('Backend URL should be correctly formatted', () {
      expect(service.getBackendUrl(), 'http://192.168.100.2:8000');
    });

    test('Backend URL should use default values when not provided', () {
      final defaultService = PredictionService();
      expect(
        defaultService.getBackendUrl(),
        'http://${PredictionService.defaultIp}:${PredictionService.defaultPort}',
      );
    });

    test('PredictionResult should parse from JSON correctly', () {
      final json = {
        'action': 'hello',
        'confidence': 0.95,
      };

      final result = PredictionResult.fromJson(json);

      expect(result.action, 'hello');
      expect(result.confidence, 0.95);
    });

    test('PredictionResult should handle integer confidence', () {
      final json = {
        'action': 'goodbye',
        'confidence': 1,
      };

      final result = PredictionResult.fromJson(json);

      expect(result.action, 'goodbye');
      expect(result.confidence, 1.0);
    });

    // Note: Async validation tests are removed because they require network connectivity
    // In a real-world scenario, you would mock the HTTP client to test these properly
  });
}
