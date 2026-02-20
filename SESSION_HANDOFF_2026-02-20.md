# SignSpeak Session Handoff (2026-02-20)

Use this file to resume work quickly in the next session.

## Project Direction (Locked)
- Keep `front-end-mobile-2` as primary mobile app (frame upload -> backend extraction).
- Do not switch to React Native now.
- Focus is **max accuracy** first, then performance tuning.

## Implemented in This Session

### 1) Model switch (Baseline/Augmented) for mobile V2
- Mobile UI has runtime model toggle (Baseline/Augmented), default is Baseline on app launch.
- Mobile sends selected model per request to backend.
- Backend accepts `POST /predict-frames?model=baseline|augmented`.
- Backend returns `model_used` in response.

Key files:
- `front-end-mobile-2/lib/main.dart`
- `front-end-mobile-2/lib/services/prediction_service.dart`
- `ml-pipeline/ml_pipeline_data_collection/api_server.py`
- `front-end-mobile-2/README.md`

### 2) Data collection camera selector
- Added camera index dropdown + rescan in collector GUI.
- Added fallback camera open logic for Windows virtual cameras (DroidCam compatibility).

Key file:
- `ml-pipeline/ml_pipeline_data_collection/collect_data_gui.py`

## Data Collection Decision
- Collect additional data with **mobile camera feed** (domain gap reduction).
- Target discussed: about +30 sequences per sign (or prioritize weak signs first if time is tight).

## Orientation / Hand Mapping Rules (Critical)
- Record only when preview is upright (not sideways).
- Keep one fixed mirror/flip setting for the entire batch.
- Left/right hand landmark colors are expected to be different; use this as quick sanity check.
- If horizontal flip matches previous dataset handedness, keep it consistent end-to-end.

## DroidCam Notes
- AVC/H.264 is acceptable (MJPEG not required).
- Consistency matters more than codec.

## ml-pipeline Cleanup Performed
- Non-essential items moved to:
  - `ml-pipeline/ml_pipeline_data_collection/_archive_cleanup_2026-02-20/`
- Active root kept focused on:
  - collection, training, API, minimal inference, baseline/augmented artifacts.

## Tomorrow Quick Start
1. Start DroidCam phone + PC client.
2. Open collector:
   - `python ml-pipeline/ml_pipeline_data_collection/collect_data_gui.py`
3. Click **Rescan Cameras**, select DroidCam index.
4. Confirm upright preview + correct left/right color mapping.
5. Start recording new sequences.

## Optional Next Improvements
- Add an explicit LH/RH detection text overlay in collector preview for faster verification.
- Add a short script to export a mobile-vs-laptop collection report for FYP documentation.
