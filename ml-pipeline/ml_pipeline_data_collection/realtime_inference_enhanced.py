# realtime_inference_enhanced.py
"""
Enhanced Real-Time Inference with Model Selection

Features:
- Command-line option to select which model to test
- On-screen display of active model
- Performance metrics tracking
- Easy model comparison
"""

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import time
import mediapipe as mp
from mediapipe_utils import mediapipe_detection, extract_keypoints, draw_landmarks
import argparse
from datetime import datetime

from actions_config import load_actions, SEQUENCE_LENGTH, PREDICTION_THRESHOLD






def get_model_info(model_path):
    """Get model file metadata."""
    if not os.path.exists(model_path):
        return None
    
    stat = os.stat(model_path)
    size_mb = stat.st_size / (1024 * 1024)
    modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
    
    return {
        'path': model_path,
        'size_mb': size_mb,
        'modified': modified
    }


def draw_model_info(image, model_name, model_info, fps, total_predictions, correct_predictions):
    """Draw model information overlay on frame."""
    h, w, _ = image.shape
    
    # Model info panel (top-left)
    panel_height = 180
    panel_width = 400
    overlay = image.copy()
    
    # Semi-transparent background
    cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (40, 40, 40), -1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
    
    # Model name (highlighted)
    cv2.rectangle(image, (5, 5), (panel_width - 5, 40), (58, 175, 169), -1)
    cv2.putText(image, f"MODEL: {model_name.upper()}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Model details
    y_offset = 60
    line_height = 25
    
    details = [
        f"Size: {model_info['size_mb']:.2f} MB",
        f"Modified: {model_info['modified']}",
        f"FPS: {fps:.1f}",
        f"Accuracy: {correct_predictions}/{total_predictions}" if total_predictions > 0 else "Accuracy: N/A",
    ]
    
    for i, detail in enumerate(details):
        cv2.putText(image, detail, (15, y_offset + i * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Instructions (bottom-left)
    instructions = [
        "[Q] Quit",
        "[R] Reset Stats",
        "[SPACE] Mark Correct",
        "[X] Mark Wrong"
    ]
    
    inst_y = h - 120
    for i, inst in enumerate(instructions):
        cv2.putText(image, inst, (10, inst_y + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return image


def prob_viz(res, actions, input_frame, predicted_action, confidence):
    """Draw probability visualization."""
    output = input_frame.copy()
    h, w, _ = output.shape
    
    # Prediction box (top-center)
    box_width = 500
    box_x = (w - box_width) // 2
    
    # Background
    cv2.rectangle(output, (box_x, 0), (box_x + box_width, 60), (245, 117, 16), -1)
    
    # Predicted word
    text = predicted_action.upper().replace("_", " ")
    cv2.putText(output, text, (box_x + 20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Confidence
    conf_text = f"{confidence:.0%}"
    cv2.putText(output, conf_text, (box_x + box_width - 100, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Probability bars (right side)
    bar_x = w - 250
    bar_y_start = 80
    
    for i, action in enumerate(actions):
        prob = res[i]
        
        # Bar background
        cv2.rectangle(output, (bar_x, bar_y_start + i * 35), 
                     (bar_x + 200, bar_y_start + i * 35 + 25), (60, 60, 60), -1)
        
        # Bar fill
        bar_length = int(prob * 200)
        color = (0, 255, 0) if action == predicted_action else (100, 100, 100)
        cv2.rectangle(output, (bar_x, bar_y_start + i * 35),
                     (bar_x + bar_length, bar_y_start + i * 35 + 25), color, -1)
        
        # Action name
        cv2.putText(output, f"{action}: {prob:.2f}", 
                   (bar_x + 5, bar_y_start + i * 35 + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Real-time PSL sign recognition')
    parser.add_argument(
        '--model',
        type=str,
        default='action_model.h5',
        help='Model file to use (default: action_model.h5)'
    )
    parser.add_argument(
        '--baseline',
        action='store_true',
        help='Use baseline model (action_model_baseline.h5)'
    )
    parser.add_argument(
        '--augmented',
        action='store_true',
        help='Use augmented model (action_model_augmented.h5)'
    )
    
    args = parser.parse_args()
    
    # Determine which model to use
    if args.baseline:
        model_path = "action_model_baseline.h5"
        encoder_path = "label_encoder_baseline.pkl"
        model_name = "Baseline"
    elif args.augmented:
        model_path = "action_model_augmented.h5"
        encoder_path = "label_encoder_augmented.pkl"
        model_name = "Augmented"
    else:
        model_path = args.model
        encoder_path = "label_encoder.pkl"
        model_name = os.path.basename(model_path).replace('.h5', '')
    
    # Get model info
    model_info = get_model_info(model_path)
    if model_info is None:
        print(f"❌ Error: Model file not found: {model_path}")
        print("\nAvailable options:")
        print("  --baseline   : Use action_model_baseline.h5")
        print("  --augmented  : Use action_model_augmented.h5")
        print("  --model PATH : Use custom model file")
        return
    
    # Load actions
    actions = load_actions()
    
    # Display startup info
    print("=" * 60)
    print("🎥 SignSpeak - Enhanced Real-Time Inference")
    print("=" * 60)
    print(f"\n📊 Model Information:")
    print(f"   Name: {model_name}")
    print(f"   Path: {model_path}")
    print(f"   Size: {model_info['size_mb']:.2f} MB")
    print(f"   Modified: {model_info['modified']}")
    print(f"\n🎯 Actions: {', '.join(actions)}")
    print(f"\n⌨️  Keyboard Controls:")
    print(f"   [Q]       - Quit")
    print(f"   [R]       - Reset statistics")
    print(f"   [SPACE]   - Mark last prediction as correct")
    print(f"   [X]       - Mark last prediction as wrong")
    print("\n▶️  Starting inference...")
    print("=" * 60)
    
    # Load model + label encoder
    model = load_model(model_path)
    le = joblib.load(encoder_path)
    
    # State variables
    sequence = []
    sentence = []
    predictions = []
    
    # Performance tracking
    total_predictions = 0
    correct_predictions = 0
    last_prediction = None
    
    # FPS tracking
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    cap = cv2.VideoCapture(0)
    
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame. Exiting...")
                break
            
            # FPS calculation
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - start_time)
                start_time = time.time()
            
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_landmarks(image, results)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)
            
            current_prediction = None
            current_confidence = 0
            
            if len(sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(sequence, axis=0)
                res = model.predict(input_data, verbose=0)[0]
                predicted_index = np.argmax(res)
                predicted_action = le.inverse_transform([predicted_index])[0]
                confidence = res[predicted_index]
                
                current_prediction = predicted_action
                current_confidence = confidence
                
                # Append last predictions list for smoothing
                predictions.append(predicted_index)
                if len(predictions) > 10:
                    predictions.pop(0)
                
                # Only accept if stable and above threshold
                if confidence >= PREDICTION_THRESHOLD and predictions.count(predicted_index) > 6:
                    if len(sentence) == 0 or sentence[-1] != predicted_action:
                        sentence.append(predicted_action)
                        last_prediction = predicted_action
                        total_predictions += 1
                
                # Keep last 5 words max
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                # Visualize probabilities
                image = prob_viz(res, actions, image, predicted_action, confidence)
            else:
                # Show "warming up" message
                cv2.putText(image, "Warming up...", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Draw model info overlay
            image = draw_model_info(image, model_name, model_info, fps, 
                                   total_predictions, correct_predictions)
            
            # Show current sentence
            if sentence:
                sentence_text = " ".join(sentence)
                # Background for sentence
                cv2.rectangle(image, (0, image.shape[0] - 50), 
                            (image.shape[1], image.shape[0]), (20, 20, 20), -1)
                cv2.putText(image, sentence_text, (10, image.shape[0] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
            
            cv2.imshow(f'SignSpeak - {model_name} Model', image)
            
            # Keyboard controls
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset statistics
                total_predictions = 0
                correct_predictions = 0
                sentence = []
                print("\n📊 Statistics reset")
            elif key == ord(' ') and last_prediction:
                # Mark correct
                correct_predictions += 1
                print(f"✅ Marked '{last_prediction}' as CORRECT ({correct_predictions}/{total_predictions})")
            elif key == ord('x') and last_prediction:
                # Mark wrong (don't increment correct)
                print(f"❌ Marked '{last_prediction}' as WRONG ({correct_predictions}/{total_predictions})")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print("\n" + "=" * 60)
    print("📊 Final Statistics")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Marked Correct: {correct_predictions}")
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Accuracy: {accuracy:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
