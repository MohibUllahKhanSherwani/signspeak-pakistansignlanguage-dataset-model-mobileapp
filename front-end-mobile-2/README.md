# SignSpeak Mobile V2

Flutter mobile client for frame-upload sign prediction.

## Runtime Model Switch

The app now supports runtime switching between:

- `Baseline`
- `Augmented`

Use the top segmented control in the header before tapping **Predict**.

Behavior:

- Default model on app launch: `Baseline`
- Selection is **not persisted** across restarts
- Toggle is disabled while prediction is in progress

## Backend API Contract

Prediction requests are sent to:

- `POST /predict-frames?model=baseline`
- `POST /predict-frames?model=augmented`

`model` query parameter rules:

- Allowed: `baseline`, `augmented`
- Omitted value defaults to `baseline`

Expected response includes:

- `action`
- `confidence`
- `all_probabilities`
- `model_used`
- `processing_time_ms`
- `frames_processed`
- `hands_detected`

## Run

```bash
flutter pub get
flutter run
```
