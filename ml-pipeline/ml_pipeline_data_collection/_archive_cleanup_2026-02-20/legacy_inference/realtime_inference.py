# realtime_inference.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import time
import mediapipe as mp
from mediapipe_utils import mediapipe_detection, extract_keypoints, draw_landmarks

from actions_config import load_actions, SEQUENCE_LENGTH, PREDICTION_THRESHOLD, DATA_PATH




def prob_viz(res, actions, input_frame):
    # Draw probability bars on image and return it
    output = input_frame.copy()
    h, w, _ = output.shape
    # Draw bars for each action
    for i, action in enumerate(actions):
        prob = res[i]
        text = f"{action}: {prob:.2f}"
        cv2.putText(output, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return output

def main():
    actions = load_actions()
    print("Actions:", actions)
    # load model + label encoder
    model = load_model("action_model.h5")
    le = joblib.load("label_encoder.pkl")

    sequence = []
    sentence = []
    predictions = []
    cap = cv2.VideoCapture(0)
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            image, results = mediapipe_detection(frame, holistic)
            # draw landmarks
            draw_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            if len(sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(sequence, axis=0)  # shape (1, seq_len, features)
                res = model.predict(input_data)[0]            # shape (n_actions,)
                predicted_index = np.argmax(res)
                predicted_action = le.inverse_transform([predicted_index])[0]
                confidence = res[predicted_index]

                # Append last predictions list for smoothing
                predictions.append(predicted_index)
                if len(predictions) > 10:
                    predictions.pop(0)

                # Only accept if stable and above threshold
                if confidence >= PREDICTION_THRESHOLD and predictions.count(predicted_index) > 6:
                    if len(sentence) == 0 or sentence[-1] != predicted_action:
                        sentence.append(predicted_action)
                # Keep last 5 words max
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Visualize
                image = prob_viz(res, actions, image)
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, " ".join(sentence), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow('SignSpeak - Real Time', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
