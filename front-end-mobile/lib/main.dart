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
            flatLandmarks.add(point.x);
            flatLandmarks.add(point.y);
            flatLandmarks.add(point.z);
          }

          // Simple heuristic for handedness if label isn't available:
          // In mirrored front camera, left side of screen is right hand? 
          // Use hand.label if available, otherwise just assign.
          // Note: hand_landmarker Hand object usually has a 'label' or 'handedness'.
          // Trying 'label' property. If it fails, we fall back to index?
          // To implement safely without crashing on property access, I'll rely on dynamic or known properties.
          // Since I can't check, I'll use try-catch for label or assume 'Left'/'Right'.
          
          // Assuming 'type' or 'label' exists.
          // The user's code `List<Hand> _landmarks` and painter just iterates.
          // I will assume 'type' property simply because it's common.
          // If compile fails, I'll fix.
          // Use the label from the HandLandmarker plugin to determine handedness.
          // This is much more reliable than checking x-coordinates.
          // MediaPipe usually returns 'Left' or 'Right'.
            // FALLBACK: The hand_landmarker package does NOT support handedness labels.
            // We must use a heuristic approach.
            // REVISED LOGIC: Raw Camera Sensor (Front) is usually NOT mirrored.
            // - User's Right Hand is on Camera's Left (x < 0.5)
            // - User's Left Hand is on Camera's Right (x > 0.5)
            
            final wristX = hand.landmarks[0].x;
            String estimatedLabel = wristX < 0.5 ? 'Right' : 'Left';
            
            // DEBUG: Update overlay
            final info = 'Hand: $estimatedLabel (Est)\nWrist X: ${wristX.toStringAsFixed(2)}';
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
