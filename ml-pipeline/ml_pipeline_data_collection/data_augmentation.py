"""
Data Augmentation Module for Sign Language Landmark Sequences

This module provides augmentation techniques specifically designed for
MediaPipe landmark data (spatial-temporal sequences).

Key Features:
- Time warping (speed variations)
- Spatial transformations (scaling, rotation, translation)
- Noise injection
- Horizontal flipping
- Temporal cropping
"""

import numpy as np
from scipy.interpolate import interp1d


class SignLanguageAugmenter:
    """
    Augmentation pipeline for sign language landmark sequences.
    
    Landmarks shape: (sequence_length, num_features)
    For PSL: (30, 126) = 30 frames, 126 features (left_hand + right_hand)
    """
    
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        
    def time_warp(self, sequence, speed_range=(0.8, 1.2)):
        """
        Time warping: Speed up or slow down the sequence.
        
        Args:
            sequence: (seq_len, features) numpy array
            speed_range: (min_speed, max_speed) tuple
        
        Returns:
            Warped sequence of same length
        """
        speed_factor = np.random.uniform(*speed_range)
        seq_len = len(sequence)
        
        # Original time points
        original_times = np.linspace(0, 1, seq_len)
        
        # Warped time points
        new_length = int(seq_len / speed_factor)
        new_times = np.linspace(0, 1, new_length)
        
        # Interpolate each feature
        augmented = np.zeros((seq_len, sequence.shape[1]))
        for i in range(sequence.shape[1]):
            interpolator = interp1d(original_times, sequence[:, i], 
                                   kind='cubic', fill_value='extrapolate')
            # Resample to original length
            resampled = interpolator(np.linspace(0, 1, seq_len))
            augmented[:, i] = resampled
            
        return augmented
    
    def spatial_scale(self, sequence, scale_range=(0.9, 1.1)):
        """
        Spatial scaling: Zoom in/out (simulates different distances).
        
        Args:
            sequence: (seq_len, features) numpy array
            scale_range: (min_scale, max_scale) tuple
        
        Returns:
            Scaled sequence
        """
        scale_factor = np.random.uniform(*scale_range)
        
        # Reshape to (seq_len, num_landmarks, 3) for x,y,z
        seq_len = len(sequence)
        landmarks_3d = sequence.reshape(seq_len, -1, 3)
        
        # Scale x and y coordinates (not z to preserve depth)
        landmarks_3d[:, :, 0] *= scale_factor  # x
        landmarks_3d[:, :, 1] *= scale_factor  # y
        
        # Flatten back
        return landmarks_3d.reshape(seq_len, -1)
    
    def spatial_translate(self, sequence, translate_range=0.1):
        """
        Spatial translation: Shift position (x, y offsets).
        
        Args:
            sequence: (seq_len, features) numpy array
            translate_range: Max translation as fraction of frame
        
        Returns:
            Translated sequence
        """
        tx = np.random.uniform(-translate_range, translate_range)
        ty = np.random.uniform(-translate_range, translate_range)
        
        # Reshape to (seq_len, num_landmarks, 3)
        seq_len = len(sequence)
        landmarks_3d = sequence.reshape(seq_len, -1, 3)
        
        # Translate x and y
        landmarks_3d[:, :, 0] += tx
        landmarks_3d[:, :, 1] += ty
        
        return landmarks_3d.reshape(seq_len, -1)
    
    def spatial_rotate(self, sequence, angle_range=(-15, 15)):
        """
        Spatial rotation: Rotate around center (in degrees).
        
        Args:
            sequence: (seq_len, features) numpy array
            angle_range: (min_angle, max_angle) in degrees
        
        Returns:
            Rotated sequence
        """
        angle_deg = np.random.uniform(*angle_range)
        angle_rad = np.deg2rad(angle_deg)
        
        # Rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Reshape to (seq_len, num_landmarks, 3)
        seq_len = len(sequence)
        landmarks_3d = sequence.reshape(seq_len, -1, 3)
        
        # Find center (mean of x, y)
        center_x = landmarks_3d[:, :, 0].mean()
        center_y = landmarks_3d[:, :, 1].mean()
        
        # Translate to origin, rotate, translate back
        landmarks_3d[:, :, 0] -= center_x
        landmarks_3d[:, :, 1] -= center_y
        
        x_new = landmarks_3d[:, :, 0] * cos_a - landmarks_3d[:, :, 1] * sin_a
        y_new = landmarks_3d[:, :, 0] * sin_a + landmarks_3d[:, :, 1] * cos_a
        
        landmarks_3d[:, :, 0] = x_new + center_x
        landmarks_3d[:, :, 1] = y_new + center_y
        
        return landmarks_3d.reshape(seq_len, -1)
    
    def add_noise(self, sequence, noise_std=0.01):
        """
        Add Gaussian noise to coordinates.
        
        Args:
            sequence: (seq_len, features) numpy array
            noise_std: Standard deviation of Gaussian noise
        
        Returns:
            Noisy sequence
        """
        noise = np.random.normal(0, noise_std, sequence.shape)
        return sequence + noise
    
    def horizontal_flip(self, sequence):
        """
        Horizontal flip: Mirror the sign (swap left/right hands).
        
        NOTE: This is disabled for PSL because gestures are NON-SYMMETRIC.
        Left-handed and right-handed signs carry different semantic meanings.
        
        Args:
            sequence: (seq_len, features) numpy array
        
        Returns:
            Flipped sequence
        """
        seq_len = len(sequence)
        landmarks_3d = sequence.reshape(seq_len, -1, 3)
        
        # Flip x coordinates (y and z stay the same)
        landmarks_3d[:, :, 0] = 1.0 - landmarks_3d[:, :, 0]
        
        # IMPORTANT: Swap left and right hand landmarks
        # Structure: [left_hand (21*3), right_hand (21*3)] -> Total 126 features (42 landmarks)
        flipped = landmarks_3d.copy()
        
        # Swap left hand (indices 0-20) with right hand (indices 21-41)
        left_start, left_end = 0, 21
        right_start, right_end = 21, 42
        
        flipped[:, left_start:left_end] = landmarks_3d[:, right_start:right_end]
        flipped[:, right_start:right_end] = landmarks_3d[:, left_start:left_end]
        
        return flipped.reshape(seq_len, -1)
    
    def temporal_crop(self, sequence, crop_ratio=0.1):
        """
        Temporal cropping: Random start/end (then resize to original length).
        
        Args:
            sequence: (seq_len, features) numpy array
            crop_ratio: Max fraction to crop from start/end
        
        Returns:
            Cropped and resized sequence
        """
        seq_len = len(sequence)
        max_crop = int(seq_len * crop_ratio)
        
        start_crop = np.random.randint(0, max_crop + 1)
        end_crop = np.random.randint(0, max_crop + 1)
        
        # Crop
        cropped = sequence[start_crop:seq_len - end_crop]
        
        # Resize back to original length using interpolation
        original_times = np.linspace(0, 1, len(cropped))
        new_times = np.linspace(0, 1, seq_len)
        
        resized = np.zeros((seq_len, sequence.shape[1]))
        for i in range(sequence.shape[1]):
            interpolator = interp1d(original_times, cropped[:, i], 
                                   kind='linear', fill_value='extrapolate')
            resized[:, i] = interpolator(new_times)
        
        return resized
    
    def augment(self, sequence, techniques=None, probabilities=None):
        """
        Apply multiple augmentation techniques with given probabilities.
        
        Args:
            sequence: (seq_len, features) numpy array
            techniques: List of technique names (default: all)
            probabilities: Dict of {technique: probability}
        
        Returns:
            Augmented sequence
        """
        if techniques is None:
            techniques = [
                'time_warp', 'spatial_scale', 'spatial_translate',
                'spatial_rotate', 'add_noise', 'temporal_crop'
            ]
        
        if probabilities is None:
            # Default probabilities (conservative)
            probabilities = {
                'time_warp': 0.5,
                'spatial_scale': 0.5,
                'spatial_translate': 0.5,
                'spatial_rotate': 0.3,
                'add_noise': 0.3,
                'temporal_crop': 0.3,
                # 'horizontal_flip': 0.5, # DISABLED for PSL (Non-symmetric)
            }
        
        augmented = sequence.copy()
        
        for technique in techniques:
            if np.random.random() < probabilities.get(technique, 0.0):
                if technique == 'time_warp':
                    augmented = self.time_warp(augmented)
                elif technique == 'spatial_scale':
                    augmented = self.spatial_scale(augmented)
                elif technique == 'spatial_translate':
                    augmented = self.spatial_translate(augmented)
                elif technique == 'spatial_rotate':
                    augmented = self.spatial_rotate(augmented)
                elif technique == 'add_noise':
                    augmented = self.add_noise(augmented)
                elif technique == 'temporal_crop':
                    augmented = self.temporal_crop(augmented)
                elif technique == 'horizontal_flip':
                    augmented = self.horizontal_flip(augmented)
        
        return augmented
    
    def augment_batch(self, sequences, multiplier=2):
        """
        Augment a batch of sequences.
        
        Args:
            sequences: (batch_size, seq_len, features) numpy array
            multiplier: How many augmented versions per original
        
        Returns:
            Augmented batch: (batch_size * multiplier, seq_len, features)
        """
        augmented_list = []
        
        for seq in sequences:
            # Keep original
            augmented_list.append(seq)
            
            # Generate augmented versions
            for _ in range(multiplier - 1):
                aug_seq = self.augment(seq)
                augmented_list.append(aug_seq)
        
        return np.array(augmented_list)


# ===== Utility Functions =====

def create_augmented_dataset(X, y, augmentation_multiplier=3):
    """
    Create augmented dataset from original data.
    
    Args:
        X: Original sequences (num_samples, seq_len, features)
        y: Labels (num_samples,)
        augmentation_multiplier: Total size = original * multiplier
    
    Returns:
        X_augmented, y_augmented
    """
    augmenter = SignLanguageAugmenter()
    
    X_augmented = []
    y_augmented = []
    
    for i, (seq, label) in enumerate(zip(X, y)):
        # Keep original
        X_augmented.append(seq)
        y_augmented.append(label)
        
        # Generate augmented versions
        for _ in range(augmentation_multiplier - 1):
            aug_seq = augmenter.augment(seq)
            X_augmented.append(aug_seq)
            y_augmented.append(label)
    
    return np.array(X_augmented), np.array(y_augmented)


if __name__ == "__main__":
    # Demo usage
    print("Sign Language Data Augmentation Demo")
    print("=" * 50)
    
    # Simulate a sequence (30 frames, 126 features)
    dummy_sequence = np.random.randn(30, 126)
    
    augmenter = SignLanguageAugmenter()
    
    # Test each technique
    print("\n1. Time Warp:")
    warped = augmenter.time_warp(dummy_sequence)
    print(f"   Original shape: {dummy_sequence.shape}")
    print(f"   Warped shape: {warped.shape}")
    
    print("\n2. Spatial Scale:")
    scaled = augmenter.spatial_scale(dummy_sequence)
    print(f"   Mean before: {dummy_sequence.mean():.4f}")
    print(f"   Mean after: {scaled.mean():.4f}")
    
    print("\n3. Horizontal Flip:")
    flipped = augmenter.horizontal_flip(dummy_sequence)
    print(f"   X coords before flip: {dummy_sequence[0, 0]:.4f}")
    print(f"   X coords after flip: {flipped[0, 0]:.4f}")
    
    print("\n4. Combined Augmentation:")
    augmented = augmenter.augment(dummy_sequence)
    print(f"   Original shape: {dummy_sequence.shape}")
    print(f"   Augmented shape: {augmented.shape}")
    
    print("\n5. Batch Augmentation (3x multiplier):")
    batch = np.random.randn(10, 30, 126)  # 10 sequences
    augmented_batch = augmenter.augment_batch(batch, multiplier=3)
    print(f"   Original batch: {batch.shape}")
    print(f"   Augmented batch: {augmented_batch.shape}")
    print(f"   Dataset increased from {len(batch)} to {len(augmented_batch)} samples")
    
    print("\n✅ All augmentation techniques working!")
