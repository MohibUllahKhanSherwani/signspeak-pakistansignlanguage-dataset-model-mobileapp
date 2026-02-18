import 'package:flutter_test/flutter_test.dart';
import 'package:sign_speak_mob_app/services/landmark_manager.dart';

void main() {
  group('LandmarkManager Tests', () {
    late LandmarkManager manager;

    setUp(() {
      manager = LandmarkManager();
    });

    test('Buffer should be empty initially', () {
      expect(manager.currentFrameCount, 0);
      expect(manager.isBufferFull, false);
      expect(manager.getBuffer(), null);
    });

    test('Adding a frame should increase frame count', () {
      final leftHand = List<double>.filled(63, 1.0);
      final rightHand = List<double>.filled(63, 2.0);

      manager.addFrame(
        leftHandLandmarks: leftHand,
        rightHandLandmarks: rightHand,
      );

      expect(manager.currentFrameCount, 1);
      expect(manager.isBufferFull, false);
    });

    test('Buffer should be full after 60 frames', () {
      final leftHand = List<double>.filled(63, 1.0);
      final rightHand = List<double>.filled(63, 2.0);

      for (int i = 0; i < 60; i++) {
        manager.addFrame(
          leftHandLandmarks: leftHand,
          rightHandLandmarks: rightHand,
        );
      }

      expect(manager.currentFrameCount, 60);
      expect(manager.isBufferFull, true);
    });

    test('Buffer should maintain exactly 60 frames', () {
      final leftHand = List<double>.filled(63, 1.0);
      final rightHand = List<double>.filled(63, 2.0);

      // Add 65 frames
      for (int i = 0; i < 65; i++) {
        manager.addFrame(
          leftHandLandmarks: leftHand,
          rightHandLandmarks: rightHand,
        );
      }

      expect(manager.currentFrameCount, 60);
      expect(manager.isBufferFull, true);
    });

    test('Each frame should contain 126 values', () {
      final leftHand = List<double>.filled(63, 1.0);
      final rightHand = List<double>.filled(63, 2.0);

      for (int i = 0; i < 60; i++) {
        manager.addFrame(
          leftHandLandmarks: leftHand,
          rightHandLandmarks: rightHand,
        );
      }

      final buffer = manager.getBuffer();
      expect(buffer, isNotNull);
      expect(buffer!.length, 60);
      
      for (final frame in buffer) {
        expect(frame.length, 126);
      }
    });

    test('Missing left hand should be padded with zeros', () {
      final rightHand = List<double>.filled(63, 2.0);

      manager.addFrame(
        leftHandLandmarks: null,
        rightHandLandmarks: rightHand,
      );

      for (int i = 1; i < 60; i++) {
        manager.addFrame(
          leftHandLandmarks: List<double>.filled(63, 1.0),
          rightHandLandmarks: rightHand,
        );
      }

      final buffer = manager.getBuffer();
      expect(buffer, isNotNull);
      
      // First frame should have zeros for left hand (positions 0-62)
      final firstFrame = buffer!.first;
      for (int i = 0; i < 63; i++) {
        expect(firstFrame[i], 0.0);
      }
      
      // Right hand should have values (positions 63-125)
      for (int i = 63; i < 126; i++) {
        expect(firstFrame[i], 2.0);
      }
    });

    test('Missing right hand should be padded with zeros', () {
      final leftHand = List<double>.filled(63, 1.0);

      manager.addFrame(
        leftHandLandmarks: leftHand,
        rightHandLandmarks: null,
      );

      for (int i = 1; i < 60; i++) {
        manager.addFrame(
          leftHandLandmarks: leftHand,
          rightHandLandmarks: List<double>.filled(63, 2.0),
        );
      }

      final buffer = manager.getBuffer();
      expect(buffer, isNotNull);
      
      // First frame left hand should have values (positions 0-62)
      final firstFrame = buffer!.first;
      for (int i = 0; i < 63; i++) {
        expect(firstFrame[i], 1.0);
      }
      
      // Right hand should have zeros (positions 63-125)
      for (int i = 63; i < 126; i++) {
        expect(firstFrame[i], 0.0);
      }
    });

    test('Clear should reset the buffer', () {
      final leftHand = List<double>.filled(63, 1.0);
      final rightHand = List<double>.filled(63, 2.0);

      for (int i = 0; i < 60; i++) {
        manager.addFrame(
          leftHandLandmarks: leftHand,
          rightHandLandmarks: rightHand,
        );
      }

      expect(manager.isBufferFull, true);

      manager.clear();

      expect(manager.currentFrameCount, 0);
      expect(manager.isBufferFull, false);
      expect(manager.getBuffer(), null);
    });

    test('getStats should return correct statistics', () {
      final stats = manager.getStats();
      
      expect(stats['bufferSize'], 0);
      expect(stats['isFull'], false);
      expect(stats['totalFrames'], 60);
      expect(stats['landmarksPerFrame'], 126);
    });
  });
}
