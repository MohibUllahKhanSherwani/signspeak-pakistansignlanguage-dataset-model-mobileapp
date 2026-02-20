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
  cameras = await availableCameras();
  runApp(const SignSpeakApp());
}

class SignSpeakApp extends StatelessWidget {
  const SignSpeakApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SignSpeak V2',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: const Color(0xFF6C63FF),
        brightness: Brightness.dark,
        textTheme: GoogleFonts.interTextTheme(
          ThemeData(brightness: Brightness.dark).textTheme,
        ),
      ),
      home: const CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isCameraReady = false;
  bool _isServerConnected = false;
  bool _isPredicting = false;

  // Circular buffer: stores the last 60 JPEG frames
  final Queue<Uint8List> _frameBuffer = Queue<Uint8List>();
  static const int _bufferSize = AppConfig.sequenceLength; // 60
  int _framesInBuffer = 0;
  bool _isBufferFull = false;

  // Frame capture throttle
  bool _isProcessingFrame = false;
  int _frameSkipCount = 0;
  // Capture every Nth frame from the stream to hit ~30fps equivalent
  // Camera streams at 30fps natively, so we take every frame
  // But if processing is slow, we skip frames
  static const int _captureEveryN = 1;

  // Prediction result
  PredictionResult? _lastResult;
  InferenceModel _selectedModel = InferenceModel.baseline;
  String? _errorMessage;

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

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _controller?.dispose();
      _controller = null;
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _checkServer() async {
    final connected = await PredictionService.checkHealth();
    if (mounted) {
      setState(() => _isServerConnected = connected);
    }
  }

  Future<void> _initCamera() async {
    if (cameras.isEmpty) return;

    // Prefer front camera
    final frontCam = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    _controller = CameraController(
      frontCam,
      ResolutionPreset.medium, // 480p — enough for MediaPipe, saves battery
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

  /// Start the camera image stream and capture frames into the buffer.
  /// This uses startImageStream() which provides frames at camera native
  /// FPS (~30fps) without disk I/O, unlike takePicture().
  void _startImageStream() {
    if (_controller == null || !_controller!.value.isInitialized) return;

    _controller!.startImageStream((CameraImage cameraImage) {
      if (_isPredicting || _isProcessingFrame) return;

      // Throttle: skip frames if needed
      _frameSkipCount++;
      if (_frameSkipCount % _captureEveryN != 0) return;

      _isProcessingFrame = true;
      _convertAndBufferFrame(cameraImage).then((_) {
        _isProcessingFrame = false;
      });
    });
  }

  /// Convert a CameraImage (YUV420) to JPEG and add to the circular buffer.
  Future<void> _convertAndBufferFrame(CameraImage cameraImage) async {
    try {
      // Yield to UI thread
      await Future.delayed(Duration.zero);

      // Convert YUV420 to img.Image
      final image = _convertYUV420ToImage(cameraImage);
      if (image == null) return;

      // Resize to target dimensions for smaller payload
      final resized = img.copyResize(
        image,
        width: AppConfig.frameWidth,
        height: AppConfig.frameHeight,
      );

      // Encode as JPEG
      final jpeg = Uint8List.fromList(
        img.encodeJpg(resized, quality: AppConfig.jpegQuality),
      );

      // Add to circular buffer
      _frameBuffer.addLast(jpeg);
      if (_frameBuffer.length > _bufferSize) {
        _frameBuffer.removeFirst();
      }

      if (mounted) {
        setState(() {
          _framesInBuffer = _frameBuffer.length;
          _isBufferFull = _frameBuffer.length >= _bufferSize;
        });
      }
    } catch (e) {
      debugPrint('Frame convert error: $e');
    }
  }

  /// Convert YUV420 CameraImage to img.Image.
  /// This handles the NV21/YUV420SP format that Android cameras produce.
  img.Image? _convertYUV420ToImage(CameraImage cameraImage) {
    try {
      final int width = cameraImage.width;
      final int height = cameraImage.height;

      final yPlane = cameraImage.planes[0];
      final uPlane = cameraImage.planes[1];
      final vPlane = cameraImage.planes[2];

      final image = img.Image(width: width, height: height);

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int yIndex = y * yPlane.bytesPerRow + x;
          final int uvIndex =
              (y ~/ 2) * uPlane.bytesPerRow + (x ~/ 2) * uPlane.bytesPerPixel!;

          final int yValue = yPlane.bytes[yIndex];
          final int uValue = uPlane.bytes[uvIndex];
          final int vValue = vPlane.bytes[uvIndex];

          // YUV to RGB conversion
          int r = (yValue + 1.370705 * (vValue - 128)).round().clamp(0, 255);
          int g =
              (yValue - 0.337633 * (uValue - 128) - 0.698001 * (vValue - 128))
                  .round()
                  .clamp(0, 255);
          int b = (yValue + 1.732446 * (uValue - 128)).round().clamp(0, 255);

          image.setPixelRgb(x, y, r, g, b);
        }
      }

      return image;
    } catch (e) {
      debugPrint('YUV conversion error: $e');
      return null;
    }
  }

  Future<void> _predict() async {
    if (!_isBufferFull || _isPredicting) return;

    setState(() {
      _isPredicting = true;
      _errorMessage = null;
    });

    // Stop the image stream during prediction upload
    try {
      await _controller?.stopImageStream();
    } catch (_) {}

    try {
      // Snapshot the buffer
      final frames = _frameBuffer.toList();

      final result = await PredictionService.predictFromFrames(
        frames,
        model: _selectedModel,
      );

      if (mounted) {
        setState(() {
          _lastResult = result;
          _isPredicting = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = e.toString();
          _isPredicting = false;
        });
      }
    }

    // Resume the image stream
    _startImageStream();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D0D1A),
      body: SafeArea(
        child: Column(
          children: [
            // Header
            _buildHeader(),

            // Camera preview
            Expanded(child: _buildCameraPreview()),

            // Prediction result
            _buildResultCard(),

            // Predict button
            _buildPredictButton(),

            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFF6C63FF), Color(0xFF9D4EDD)],
                  ),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(
                  Icons.sign_language,
                  color: Colors.white,
                  size: 24,
                ),
              ),
              const SizedBox(width: 12),
              Text(
                'SignSpeak',
                style: GoogleFonts.inter(
                  fontSize: 22,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
              const Spacer(),
              // Server status indicator
              _buildServerBadge(),
            ],
          ),
          const SizedBox(height: 10),
          _buildModelSelector(),
        ],
      ),
    );
  }

  Widget _buildModelSelector() {
    return SizedBox(
      width: double.infinity,
      child: SegmentedButton<InferenceModel>(
        segments: const [
          ButtonSegment<InferenceModel>(
            value: InferenceModel.baseline,
            label: Text('Baseline'),
            icon: Icon(Icons.speed, size: 16),
          ),
          ButtonSegment<InferenceModel>(
            value: InferenceModel.augmented,
            label: Text('Augmented'),
            icon: Icon(Icons.auto_awesome, size: 16),
          ),
        ],
        selected: {_selectedModel},
        showSelectedIcon: false,
        onSelectionChanged: _isPredicting
            ? null
            : (selection) {
                if (selection.isEmpty) return;
                setState(() {
                  _selectedModel = selection.first;
                });
              },
      ),
    );
  }

  Widget _buildServerBadge() {
    return GestureDetector(
      onTap: _checkServer, // Tap to refresh connection
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: _isServerConnected
              ? Colors.green.withOpacity(0.15)
              : Colors.red.withOpacity(0.15),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: _isServerConnected
                ? Colors.green.withOpacity(0.4)
                : Colors.red.withOpacity(0.4),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _isServerConnected ? Colors.green : Colors.red,
              ),
            ),
            const SizedBox(width: 6),
            Text(
              _isServerConnected ? 'Connected' : 'Offline',
              style: TextStyle(
                fontSize: 12,
                color: _isServerConnected ? Colors.green : Colors.red,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (!_isCameraReady || _controller == null) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(color: Color(0xFF6C63FF)),
            SizedBox(height: 16),
            Text(
              'Initializing camera...',
              style: TextStyle(color: Colors.white54),
            ),
          ],
        ),
      );
    }

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: _isBufferFull
              ? const Color(0xFF6C63FF).withOpacity(0.6)
              : Colors.white.withOpacity(0.1),
          width: 2,
        ),
        boxShadow: _isBufferFull
            ? [
                BoxShadow(
                  color: const Color(0xFF6C63FF).withOpacity(0.2),
                  blurRadius: 20,
                  spreadRadius: 2,
                ),
              ]
            : null,
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(18),
        child: Stack(
          fit: StackFit.expand,
          children: [
            // Camera feed
            CameraPreview(_controller!),

            // Buffer fill indicator overlay
            Positioned(
              top: 12,
              left: 12,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.6),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      _isBufferFull ? Icons.check_circle : Icons.hourglass_top,
                      color: _isBufferFull ? Colors.greenAccent : Colors.amber,
                      size: 16,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      _isBufferFull
                          ? 'Ready (${_framesInBuffer}f)'
                          : 'Buffering $_framesInBuffer/$_bufferSize',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // Processing overlay
            if (_isPredicting)
              Container(
                color: Colors.black.withOpacity(0.4),
                child: const Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      CircularProgressIndicator(
                        color: Color(0xFF6C63FF),
                        strokeWidth: 3,
                      ),
                      SizedBox(height: 16),
                      Text(
                        'Processing...',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        'Running MediaPipe + LSTM on server',
                        style: TextStyle(color: Colors.white54, fontSize: 12),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard() {
    if (_errorMessage != null) {
      return Container(
        margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.red.withValues(alpha: 0.1),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.red.withValues(alpha: 0.3)),
        ),
        child: Row(
          children: [
            const Icon(Icons.error_outline, color: Colors.redAccent),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                _errorMessage!,
                style: const TextStyle(color: Colors.redAccent, fontSize: 13),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      );
    }

    if (_lastResult == null) {
      return Container(
        margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.03),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white.withOpacity(0.06)),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'Sign a word, then tap Predict',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white.withOpacity(0.4),
                fontSize: 15,
              ),
            ),
            const SizedBox(height: 8),
            _metaChip(Icons.tune, 'Selected model: ${_selectedModel.label}'),
          ],
        ),
      );
    }

    final result = _lastResult!;
    final isConfident = result.confidence >= AppConfig.confidenceThreshold;

    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: isConfident
              ? [const Color(0xFF1A1A2E), const Color(0xFF16213E)]
              : [const Color(0xFF2D1B1B), const Color(0xFF1A1A2E)],
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isConfident
              ? const Color(0xFF6C63FF).withValues(alpha: 0.3)
              : Colors.orange.withValues(alpha: 0.3),
        ),
      ),
      child: Column(
        children: [
          // Action name
          Text(
            result.action.toUpperCase().replaceAll('_', ' '),
            style: GoogleFonts.inter(
              fontSize: 28,
              fontWeight: FontWeight.w800,
              color: isConfident ? const Color(0xFF6C63FF) : Colors.orange,
              letterSpacing: 1.5,
            ),
          ),
          const SizedBox(height: 8),

          // Confidence bar
          Row(
            children: [
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: result.confidence,
                    minHeight: 6,
                    backgroundColor: Colors.white.withValues(alpha: 0.1),
                    valueColor: AlwaysStoppedAnimation(
                      isConfident ? const Color(0xFF6C63FF) : Colors.orange,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Text(
                '${(result.confidence * 100).toStringAsFixed(1)}%',
                style: TextStyle(
                  color: isConfident ? Colors.white : Colors.orange,
                  fontWeight: FontWeight.w600,
                  fontSize: 14,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),

          // Meta info
          Wrap(
            alignment: WrapAlignment.center,
            spacing: 12,
            runSpacing: 8,
            children: [
              _metaChip(Icons.timer, '${result.processingTimeMs}ms'),
              _metaChip(
                Icons.back_hand,
                '${result.handsDetected}/${result.framesProcessed} hands',
              ),
              _metaChip(
                Icons.tune,
                'model: ${_toModelLabel(result.modelUsed)}',
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _metaChip(IconData icon, String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: Colors.white38),
          const SizedBox(width: 4),
          Text(
            text,
            style: const TextStyle(fontSize: 11, color: Colors.white54),
          ),
        ],
      ),
    );
  }

  Widget _buildPredictButton() {
    final canPredict = _isBufferFull && !_isPredicting && _isServerConnected;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: SizedBox(
        width: double.infinity,
        height: 56,
        child: ElevatedButton(
          onPressed: canPredict ? _predict : null,
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF6C63FF),
            disabledBackgroundColor: Colors.white.withOpacity(0.05),
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            elevation: canPredict ? 8 : 0,
            shadowColor: const Color(0xFF6C63FF).withOpacity(0.4),
          ),
          child: _isPredicting
              ? const SizedBox(
                  width: 24,
                  height: 24,
                  child: CircularProgressIndicator(
                    color: Colors.white,
                    strokeWidth: 2.5,
                  ),
                )
              : Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      canPredict ? Icons.auto_awesome : Icons.hourglass_top,
                      size: 22,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      canPredict
                          ? 'Predict (${_selectedModel.label})'
                          : _isServerConnected
                          ? 'Buffering...'
                          : 'Server Offline',
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
        ),
      ),
    );
  }

  String _toModelLabel(String modelKey) {
    switch (modelKey.toLowerCase()) {
      case 'augmented':
        return 'Augmented';
      case 'baseline':
      default:
        return 'Baseline';
    }
  }
}
