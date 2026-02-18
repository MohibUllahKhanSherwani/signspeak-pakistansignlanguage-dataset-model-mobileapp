import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:provider/provider.dart';
import 'package:flutter/services.dart'; // For rootBundle
import 'package:hand_landmarker/hand_landmarker.dart';
import 'providers/app_state.dart';
import 'widgets/camera_view.dart';
import 'widgets/connection_indicator.dart';

// Global camera list
late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize cameras
  try {
    cameras = await availableCameras();
  } catch (e) {
    print('Error initializing cameras: $e');
    cameras = [];
  }
  
  runApp(const SignSpeakApp());
}

class SignSpeakApp extends StatelessWidget {
  const SignSpeakApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => AppState(
        // TODO: Update with your backend IP address
        backendIp: '192.168.100.2', // CHANGE THIS TO YOUR BACKEND IP
        backendPort: 8000,
      ),
      child: MaterialApp(
        title: 'SignSpeak',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.deepPurple,
            brightness: Brightness.dark,
          ),
          useMaterial3: true,
        ),
        home: const SignLanguageRecognitionScreen(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}

class SignLanguageRecognitionScreen extends StatefulWidget {
  const SignLanguageRecognitionScreen({super.key});

  @override
  State<SignLanguageRecognitionScreen> createState() =>
      _SignLanguageRecognitionScreenState();
}

class _SignLanguageRecognitionScreenState
    extends State<SignLanguageRecognitionScreen> with WidgetsBindingObserver {
  CameraController? _cameraController;
  HandLandmarkerPlugin? _handLandmarker;
  bool _isProcessingFrame = false;
  bool _isCameraInitialized = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeApp();
  }

  Future<void> _initializeApp() async {
    // Initialize Hand Landmarker
    try {
      _handLandmarker = HandLandmarkerPlugin.create(
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        delegate: HandLandmarkerDelegate.gpu,
      );
    } catch (e) {
      print('Error initializing HandLandmarker: $e');
    }

    // Initialize camera
    await _initializeCamera();

    // Check backend connection
    if (mounted) {
      await context.read<AppState>().checkConnection();
    }
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) {
      print('No cameras available');
      return;
    }

    // Use front camera for sign language
    final camera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
      orElse: () => cameras.first,
    );

    _cameraController = CameraController(
      camera,
      ResolutionPreset.medium, // Changed to match backend default (likely 640x480)
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _cameraController!.initialize();
      
      if (!mounted) return;

      setState(() {
        _isCameraInitialized = true;
      });

      // Start image stream for landmark detection
      _cameraController!.startImageStream(_processCameraImage);
    } catch (e) {
      print('Error initializing camera: $e');
    }
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_isProcessingFrame || _handLandmarker == null || _cameraController == null) return;

    _isProcessingFrame = true;

    try {
      // Detect hands using the plugin
      // Passing orientation is required per user's example
      final hands = _handLandmarker!.detect(
        image,
        _cameraController!.description.sensorOrientation,
      );

      if (hands.isNotEmpty && mounted) {
        List<double>? leftHandLandmarks;
        List<double>? rightHandLandmarks;

        for (final hand in hands) {
          final flatLandmarks = <double>[];
          for (final point in hand.landmarks) {
            // MIRROR X-AXIS: Front camera raw frames are horizontally flipped
            // compared to the rear webcam used during Python training.
            // Applying 1.0 - x converts to rear-camera-equivalent coordinates.
            flatLandmarks.add(1.0 - point.x);
            flatLandmarks.add(point.y);
            flatLandmarks.add(point.z);
          }

          // hand_landmarker v2.2.0 does NOT expose handedness labels.
          // Use wrist X heuristic on MIRRORED coordinates (rear-camera convention):
          //   correctedWristX < 0.5 → left side of rear-cam frame → user's LEFT hand
          //   correctedWristX >= 0.5 → right side of rear-cam frame → user's RIGHT hand
          // This matches how MediaPipe Holistic classifies hands during Python training.
          final correctedWristX = 1.0 - hand.landmarks[0].x;
          String estimatedLabel = correctedWristX < 0.5 ? 'Left' : 'Right';
            
            // DEBUG: Update overlay with both raw and corrected values
            final rawX = hand.landmarks[0].x;
            final info = 'Hand: $estimatedLabel\nRaw X: ${rawX.toStringAsFixed(3)} → Corrected: ${correctedWristX.toStringAsFixed(3)}';
            print(info); // Keep console log
            context.read<AppState>().updateDebugInfo(info);
            
            if (estimatedLabel == 'Left') {
              leftHandLandmarks = flatLandmarks;
            } else {
              rightHandLandmarks = flatLandmarks;
            }
        }

        // Add to buffer
        context.read<AppState>().addLandmarkFrame(
          leftHandLandmarks: leftHandLandmarks,
          rightHandLandmarks: rightHandLandmarks,
        );
      }
    } catch (e) {
      print('Error processing image: $e');
    } finally {
      _isProcessingFrame = false;
    }
  }

  // Remove _convertCameraImage entirely as it's not needed for HandLandmarker (per Result 210)
  
  // Remove _extractHandLandmarksFromPose entirely

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _handLandmarker?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Stack(
          children: [
            // Camera View
            _isCameraInitialized
                ? CameraView(cameraController: _cameraController)
                : const Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 16),
                        Text(
                          'Initializing camera...',
                          style: TextStyle(
                            fontSize: 16,
                            color: Colors.white70,
                          ),
                        ),
                      ],
                    ),
                  ),

            // Top Bar with Connection Status and Title
            Positioned(
              top: 16,
              right: 16,
              child: Consumer<AppState>(
                builder: (context, appState, child) {
                  return ConnectionIndicator(
                    isConnected: appState.isConnected,
                    backendUrl: appState.backendUrl,
                  );
                },
              ),
            ),

            Positioned(
              top: 16,
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 10,
                  ),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        Colors.deepPurple.withOpacity(0.9),
                        Colors.purple.withOpacity(0.9),
                      ],
                    ),
                    borderRadius: BorderRadius.circular(25),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.3),
                        blurRadius: 10,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: const Text(
                    'SignSpeak',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.5,
                    ),
                  ),
                ),
              ),
            ),

            // Error Message Display
            Consumer<AppState>(
              builder: (context, appState, child) {
                if (appState.errorMessage != null) {
                  return Positioned(
                    bottom: 16,
                    left: 16,
                    right: 16,
                    child: Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.red.withOpacity(0.9),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        appState.errorMessage!,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  );
                }
                return const SizedBox.shrink();
              },
            ),
          ],
        ),
      ),
    );
  }
}
