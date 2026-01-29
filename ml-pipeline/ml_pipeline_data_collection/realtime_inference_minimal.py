# realtime_inference_minimal.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import mediapipe as mp
from mediapipe_utils import mediapipe_detection, extract_keypoints, draw_landmarks
import argparse
import os

from actions_config import load_actions, SEQUENCE_LENGTH, PREDICTION_THRESHOLD

def main():
    parser = argparse.ArgumentParser(description='Minimal SignSpeak Inference')
    parser.add_argument('--augmented', action='store_true', help='Use augmented model')
    parser.add_argument('--baseline', action='store_true', help='Use baseline model')
    args = parser.parse_args()

    # Model Selection
    model_path = "action_model_augmented.h5" if args.augmented else "action_model_baseline.h5"
    encoder_path = "label_encoder_augmented.pkl" if args.augmented else "label_encoder_baseline.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Running baseline instead.")
        model_path, encoder_path = "action_model_baseline.h5", "label_encoder_baseline.pkl"

    # Load resources
    print(f"Loading {model_path}...")
    actions = load_actions()
    model = load_model(model_path)
    le = joblib.load(encoder_path)

    # State variables
    sequence = []
    sentence = []
    predictions = []
    
    cap = cv2.VideoCapture(0)
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            # Process keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            # Instant Prediction Logic
            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                prediction_idx = np.argmax(res)
                action = le.inverse_transform([prediction_idx])[0]
                confidence = res[prediction_idx]

                # Update status and sentence
                predictions.append(prediction_idx)
                if len(predictions) > 10: predictions.pop(0)

                # Lower threshold for faster "instant" feel (changed from 6 to 3)
                if confidence > PREDICTION_THRESHOLD and predictions.count(prediction_idx) > 3:
                    if not sentence or sentence[-1] != action:
                        sentence.append(action)
                
                if len(sentence) > 5: sentence = sentence[-5:]

                # --- MINIMAL UI ---
                h, w, _ = image.shape
                
                # 1. Prediction Box (Top)
                # Just show the CURRENT word and confidence clearly
                text = f"{action.upper().replace('_', ' ')} ({confidence:.0%})"
                color = (46, 204, 113) if confidence > 0.8 else (52, 152, 219)
                
                cv2.rectangle(image, (0, 0), (w, 50), (20, 20, 20), -1)
                cv2.putText(image, text, (int(w/2) - 100, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            # 2. Sentence Box (Bottom)
            if sentence:
                cv2.rectangle(image, (0, h-60), (w, h), (44, 62, 80), -1)
                cv2.putText(image, " ".join(sentence).upper(), (20, h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('SignSpeak - Minimal Mode', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
