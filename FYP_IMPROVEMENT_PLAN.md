# Roadmap: Improving SignSpeak Recognition Accuracy

This plan outlines the steps to take your Pakistan Sign Language (PSL) project from a "prototype" to a robust, high-accuracy Final Year Project (FYP).

## Phase 1: Data Enrichment (Multi-Domain Training)
The current "Domain Gap" between your laptop webcam and mobile phone is the main cause of low accuracy. We will fix this by combining both datasets.

1.  **Update Configuration**: 
    - In `ml_pipeline_data_collection/actions_config.py`, change `NUM_SEQUENCES = 50` to `NUM_SEQUENCES = 100`.
2.  **Mobile-as-Webcam**: 
    - Use apps like **Iriun Webcam** or **DroidCam** to use your phone's camera on your PC.
3.  **Record New Batch**: 
    - Run `collect_data_gui.py`. It will see folders 0-49 already exist and automatically begin recording from folder 50.
    - Record 50 new samples per action using the phone camera.
4.  **Retrain**: 
    - Run your training script. The model will now learn features from both devices simultaneously.

## Phase 2: Feature Robustness (Landmark Normalization)
*Note: This is an advanced step. If you apply this, you must re-collect or re-process your data.*

Right now, the model is sensitive to where you are on the screen. To make it "Position Independent":
- Modify `extract_keypoints` in `mediapipe_utils.py`.
- Subtract the **Wrist Landmark** coordinates from every other point.
- This ensures the model only looks at the **shape** and **relative movement** of the hand, ignoring your distance from the camera.

## Phase 3: Model Architecture Upgrades
To impress your FYP examiners, transition from a simple LSTM to a more modern temporal architecture:
- **GRU (Gated Recurrent Unit)**: Faster to train and often better for smaller sign language datasets.
- **Attention Layers**: Add an Attention mechanism to help the model "focus" on the frames where the most important hand movement happens.
- **Bidirectional LSTM**: Allows the model to look at the sign "backwards and forwards" to better understand the full gesture.

## Phase 4: Environmental Testing
Document your testing in different environments for your final report:
- **Variable Lighting**: Test in daylight vs. evening light.
- **Backgrounds**: Test in a car, at a desk, and against a plain wall.
- **Distance**: Test from 1 meter vs. 2 meters.

---
**Current Status (Session Conclusion):**
- **Architecture**: Image-based (Option C) is fully implemented.
- **Backend**: FastAPI with automated 90° CCW rotation and horizontal mirroring is active.
- **Mobile**: Premium Flutter app (`front-end-mobile-2`) with 30fps circular buffering is functional.
- **Connection**: Confirmed working on local IP `192.168.100.2`.
