import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """
    Perform MediaPipe detection on an image.
    
    Args:
        image: BGR image from OpenCV
        model: MediaPipe model instance
        
    Returns:
        image: Original BGR image (processable)
        results: Detection results object
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results

def extract_keypoints(results):
    """
    Extract ONLY hand landmarks from MediaPipe results.
    
    Returns a concatenated array of:
    - Left Hand: 21 landmarks * 3 (x,y,z) = 63 features
    - Right Hand: 21 landmarks * 3 (x,y,z) = 63 features
    Total: 126 features
    """
    # Left hand (21 landmarks)
    lh = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()

    # Right hand (21 landmarks)
    rh = np.zeros(21 * 3)
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([lh, rh])

def draw_landmarks(image, results):
    """
    Draw only hand landmarks on the image.
    """
    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )
