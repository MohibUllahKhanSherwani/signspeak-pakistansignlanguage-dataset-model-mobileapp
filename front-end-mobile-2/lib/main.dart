import 'dart:async';
import 'dart:typed_data';
import 'dart:collection';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:google_fonts/google_fonts.dart';
import 'config.dart';
import 'services/prediction_service.dart';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Available cameras error: $e');
    cameras = [];
  }
  runApp(const SignSpeakApp());
}

class SignSpeakApp extends StatelessWidget {
  const SignSpeakApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SignSpeak Pro',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: const Color(0xFF6366F1),
        brightness: Brightness.dark,
        textTheme: GoogleFonts.outfitTextTheme(
          ThemeData(brightness: Brightness.dark).textTheme,
        ),
      ),
      home: const LandingScreen(),
    );
  }
}

/// A Premium Landing Screen for the App
class LandingScreen extends StatelessWidget {
  const LandingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF0F172A), Color(0xFF1E293B)],
          ),
        ),
        child: SafeArea(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Logo/Icon
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: const Color(0xFF6366F1).withOpacity(0.1),
                  shape: BoxShape.circle,
                  border: Border.all(color: const Color(0xFF6366F1).withOpacity(0.3)),
                ),
                child: const Icon(
                  Icons.sign_language_rounded,
                  size: 80,
                  color: Color(0xFF818CF8),
                ),
              ),
              const SizedBox(height: 40),
              Text(
                'SignSpeak',
                style: GoogleFonts.outfit(
                  fontSize: 48,
                  fontWeight: FontWeight.w800,
                  letterSpacing: -1,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 12),
              Text(
                'Bridge the gap with AI-powered\nSign Language Translation',
                textAlign: TextAlign.center,
                style: GoogleFonts.outfit(
                  fontSize: 18,
                  color: Colors.white60,
                  fontWeight: FontWeight.w400,
                ),
              ),
              const SizedBox(height: 60),
              // Get Started Button
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 40),
                child: SizedBox(
                  width: double.infinity,
                  height: 64,
                  child: ElevatedButton(
                    onPressed: () {
                      Navigator.of(context).push(
                        MaterialPageRoute(builder: (_) => const CameraScreen()),
                      );
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF6366F1),
                      foregroundColor: Colors.white,
                      elevation: 10,
                      shadowColor: const Color(0xFF6366F1).withOpacity(0.5),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Text(
                          'Start Recognition',
                          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                        ),
                        const SizedBox(width: 12),
                        const Icon(Icons.arrow_forward_rounded),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 24),
              const Text(
                'Version 2.1.0 (Mobile Optimizer)',
                style: TextStyle(color: Colors.white24, fontSize: 12),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

enum RecordingState { standby, countdown, recording, predicting, result }

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isCameraReady = false;
  bool _isServerConnected = false;
  RecordingState _state = RecordingState.standby;

  // Circular buffer: stores the target 60 JPEG frames
  final Queue<Uint8List> _frameBuffer = Queue<Uint8List>();
  static const int _targetFrames = AppConfig.sequenceLength;
  
  bool _isProcessingFrame = false;
  PredictionResult? _lastResult;
  String? _errorMessage;
  int _countdownValue = 1;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
    _checkServer();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _checkServer() async {
    final connected = await PredictionService.checkHealth();
    if (mounted) {
      setState(() => _isServerConnected = connected);
    }
  }

  Future<void> _initCamera() async {
    if (cameras.isEmpty) return;

    final frontCam = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    _controller = CameraController(
      frontCam,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _controller!.initialize();
      if (mounted) {
        setState(() => _isCameraReady = true);
        _startImageStream();
      }
    } catch (e) {
      debugPrint('Camera init error: $e');
    }
  }

  void _startImageStream() {
    if (_controller == null || !_controller!.value.isInitialized) return;

    _controller!.startImageStream((CameraImage cameraImage) {
      // ONLY buffer if we are in the recording state
      if (_state != RecordingState.recording || _isProcessingFrame) return;

      _isProcessingFrame = true;
      _convertAndBufferFrame(cameraImage).then((_) {
        _isProcessingFrame = false;
        
        // AUTO-TRIGGER prediction when buffer is full
        if (_frameBuffer.length >= _targetFrames && _state == RecordingState.recording) {
          _predict();
        }
      });
    });
  }

  Future<void> _startSequenceCapture() async {
    if (!_isServerConnected) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Wait for server connection...')),
      );
      return;
    }

    // 1. Start Countdown (0.5s as requested)
    setState(() {
      _state = RecordingState.countdown;
      _frameBuffer.clear();
      _errorMessage = null;
    });

    await Future.delayed(const Duration(milliseconds: 500));

    // 2. Start Recording
    if (mounted) {
      setState(() => _state = RecordingState.recording);
    }
  }

  Future<void> _convertAndBufferFrame(CameraImage cameraImage) async {
    try {
      final image = _convertYUV420ToImage(cameraImage);
      if (image == null) return;

      final resized = img.copyResize(
        image,
        width: AppConfig.frameWidth,
        height: AppConfig.frameHeight,
      );

      final jpeg = Uint8List.fromList(
        img.encodeJpg(resized, quality: AppConfig.jpegQuality),
      );

      _frameBuffer.addLast(jpeg);
      
      if (mounted) {
        setState(() {}); // Update progress bar
      }
    } catch (e) {
      debugPrint('Frame conversion error: $e');
    }
  }

  img.Image? _convertYUV420ToImage(CameraImage cameraImage) {
    try {
      final int width = cameraImage.width;
      final int height = cameraImage.height;
      final image = img.Image(width: width, height: height);

      final yPlane = cameraImage.planes[0];
      final uPlane = cameraImage.planes[1];
      final vPlane = cameraImage.planes[2];

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int yIndex = y * yPlane.bytesPerRow + x;
          final int uvIndex = (y ~/ 2) * uPlane.bytesPerRow + (x ~/ 2) * uPlane.bytesPerPixel!;

          final int yValue = yPlane.bytes[yIndex];
          final int uValue = uPlane.bytes[uvIndex];
          final int vValue = vPlane.bytes[uvIndex];

          int r = (yValue + 1.370705 * (vValue - 128)).round().clamp(0, 255);
          int g = (yValue - 0.337633 * (uValue - 128) - 0.698001 * (vValue - 128)).round().clamp(0, 255);
          int b = (yValue + 1.732446 * (uValue - 128)).round().clamp(0, 255);

          image.setPixelRgb(x, y, r, g, b);
        }
      }
      return image;
    } catch (e) {
      return null;
    }
  }

  Future<void> _predict() async {
    setState(() => _state = RecordingState.predicting);

    try {
      final frames = _frameBuffer.toList();
      final result = await PredictionService.predictFromFrames(frames);

      if (mounted) {
        setState(() {
          _lastResult = result;
          _state = RecordingState.result;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = e.toString();
          _state = RecordingState.standby;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F172A),
      body: SafeArea(
        child: Column(
          children: [
            _buildHeader(),
            Expanded(child: _buildCameraContainer()),
            _buildControlArea(),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(Icons.arrow_back_ios_new_rounded, color: Colors.white70),
          ),
          const SizedBox(width: 8),
          Text(
            'Recognition',
            style: GoogleFonts.outfit(fontSize: 24, fontWeight: FontWeight.bold),
          ),
          const Spacer(),
          _buildConnectionBadge(),
        ],
      ),
    );
  }

  Widget _buildConnectionBadge() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: _isServerConnected ? Colors.green.withOpacity(0.1) : Colors.red.withOpacity(0.1),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: _isServerConnected ? Colors.green.withOpacity(0.3) : Colors.red.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Container(width: 8, height: 8, decoration: BoxDecoration(shape: BoxShape.circle, color: _isServerConnected ? Colors.green : Colors.red)),
          const SizedBox(width: 6),
          Text(_isServerConnected ? 'Online' : 'Offline', style: TextStyle(fontSize: 12, color: _isServerConnected ? Colors.green : Colors.red)),
        ],
      ),
    );
  }

  Widget _buildCameraContainer() {
    if (!_isCameraReady) return const Center(child: CircularProgressIndicator());

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(28),
        border: Border.all(color: _getBorderColor(), width: 2),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(26),
        child: Stack(
          fit: StackFit.expand,
          children: [
            CameraPreview(_controller!),
            _buildOverlay(),
          ],
        ),
      ),
    );
  }

  Color _getBorderColor() {
    switch (_state) {
      case RecordingState.recording: return Colors.redAccent;
      case RecordingState.countdown: return Colors.orangeAccent;
      case RecordingState.predicting: return const Color(0xFF6366F1);
      default: return Colors.white10;
    }
  }

  Widget _buildOverlay() {
    if (_state == RecordingState.countdown) {
      return Container(
        color: Colors.black45,
        child: const Center(child: Text('READY', style: TextStyle(fontSize: 60, fontWeight: FontWeight.w900, color: Colors.orangeAccent))),
      );
    }
    
    if (_state == RecordingState.recording) {
      return Positioned(
        top: 20, right: 20,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
          decoration: BoxDecoration(color: Colors.red, borderRadius: BorderRadius.circular(8)),
          child: const Row(children: [Icon(Icons.circle, size: 12, color: Colors.white), SizedBox(width: 8), Text('REC', style: TextStyle(fontWeight: FontWeight.bold))]),
        ),
      );
    }

    if (_state == RecordingState.predicting) {
      return Container(
        color: Colors.black54,
        child: const Center(child: Column(mainAxisSize: MainAxisSize.min, children: [CircularProgressIndicator(), SizedBox(height: 16), Text('Analyzing Signs...')])),
      );
    }

    return const SizedBox.shrink();
  }

  Widget _buildControlArea() {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          if (_state == RecordingState.result && _lastResult != null) _buildResultCard(),
          if (_errorMessage != null) Text(_errorMessage!, style: const TextStyle(color: Colors.redAccent)),
          const SizedBox(height: 20),
          _buildActionButton(),
        ],
      ),
    );
  }

  Widget _buildResultCard() {
    final res = _lastResult!;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white.withOpacity(0.05), borderRadius: BorderRadius.circular(20), border: Border.all(color: Colors.white10)),
      child: Row(
        children: [
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('PREDICTION', style: TextStyle(color: Colors.white38, fontSize: 10, fontWeight: FontWeight.bold, letterSpacing: 1.2)),
              Text(res.action.toUpperCase(), style: GoogleFonts.outfit(fontSize: 32, fontWeight: FontWeight.w800, color: const Color(0xFF818CF8))),
            ]),
          ),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(color: const Color(0xFF6366F1).withOpacity(0.1), borderRadius: BorderRadius.circular(15)),
            child: Text('${(res.confidence * 100).toInt()}%', style: const TextStyle(fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton() {
    bool isWorking = _state == RecordingState.countdown || _state == RecordingState.recording || _state == RecordingState.predicting;
    
    return SizedBox(
      width: double.infinity,
      height: 70,
      child: ElevatedButton(
        onPressed: isWorking ? null : _startSequenceCapture,
        style: ElevatedButton.styleFrom(
          backgroundColor: _state == RecordingState.result ? Colors.white12 : const Color(0xFF6366F1),
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        ),
        child: _state == RecordingState.recording 
          ? Text('RECORDING ${_frameBuffer.length}/60', style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold))
          : Text(_state == RecordingState.result ? 'Record Again' : 'Start Translation', style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
      ),
    );
  }
}
