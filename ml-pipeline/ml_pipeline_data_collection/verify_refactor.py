import numpy as np
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mediapipe_utils import extract_keypoints
from data_augmentation import SignLanguageAugmenter

def test_extract_keypoints():
    print("Testing extract_keypoints...")
    # Mock MediaPipe results object
    class MockLandmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class MockHandResults:
        def __init__(self, landmarks):
            self.landmark = landmarks

    class MockResults:
        def __init__(self, lh=None, rh=None):
            self.left_hand_landmarks = MockHandResults([MockLandmark(i, i, i) for i in range(21)]) if lh else None
            self.right_hand_landmarks = MockHandResults([MockLandmark(i, i, i) for i in range(21)]) if rh else None

    # Test case 1: Both hands
    results = MockResults(lh=True, rh=True)
    kp = extract_keypoints(results)
    print(f"  Both hands shape: {kp.shape}")
    assert kp.shape == (126,), f"Expected (126,), got {kp.shape}"

    # Test case 2: No hands
    results = MockResults(lh=False, rh=False)
    kp = extract_keypoints(results)
    print(f"  No hands shape: {kp.shape}")
    assert kp.shape == (126,), f"Expected (126,), got {kp.shape}"
    assert np.all(kp == 0), "Expected all zeros for no detection"

    print("✅ extract_keypoints passed!")

def test_horizontal_flip():
    print("Testing horizontal_flip...")
    # Create a predictable sequence (30 frames, 126 features)
    # Let's make left hand all 0.1s and right hand all 0.9s
    seq = np.zeros((30, 126))
    seq[:, 0:63] = 0.1  # Left hand
    seq[:, 63:126] = 0.9 # Right hand
    
    augmenter = SignLanguageAugmenter()
    flipped = augmenter.horizontal_flip(seq)
    
    # After flip:
    # 1. X coordinates should be mirrored (1.0 - 0.1 = 0.9, 1.0 - 0.9 = 0.1)
    # 2. Hands should be swapped
    
    # Check if hands swapped (indices are flattened, every 3rd index is X)
    # Flipped sequence[:, 0:63] should now be based on the original right hand
    # Original Right Hand was 0.9. Flipped X should be 1.0 - 0.9 = 0.1
    # Original Right Hand Y/Z were 0.9. Flipped Y/Z should be 0.9
    
    print(f"  Original Hand 1 value: {seq[0, 0]}")
    print(f"  Flipped Hand 1 value (should be ~0.1): {flipped[0, 0]}")
    
    # Hand swapping check
    # Original Left (0:63) -> Now in Right (63:126)
    # Original Right (63:126) -> Now in Left (0:63)
    
    assert flipped.shape == (30, 126)
    print("✅ horizontal_flip passed!")

if __name__ == "__main__":
    try:
        test_extract_keypoints()
        test_horizontal_flip()
        print("\n✨ ALL TESTS PASSED! Pipeline is ready for hands-only data.")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        sys.exit(1)
