"""
Validate Preprocessing Parity: Flutter vs Python

Compares debug_flutter_data.npy and debug_python_data.npy to identify
coordinate system mismatches, handedness swaps, and value range issues.

Usage:
    python validate_parity.py

Prerequisite:
    1. Run Python real-time inference, press 'S' to save debug_python_data.npy
    2. Run Flutter app, perform a sign - it auto-saves debug_flutter_data.npy via api_server
"""

import numpy as np
import os
import sys


def load_debug_data(filename):
    """Load a debug .npy file if it exists."""
    if not os.path.exists(filename):
        return None
    data = np.load(filename)
    return data


def analyze_hand(data, hand_name, col_start, col_end):
    """Analyze one hand's data across all frames."""
    hand = data[:, col_start:col_end]
    present = np.any(hand != 0)
    
    if not present:
        print(f"  {hand_name}: NOT PRESENT (all zeros)")
        return
    
    # Extract x, y, z separately (stride of 3)
    x_vals = hand[:, 0::3]  # columns 0, 3, 6, ...
    y_vals = hand[:, 1::3]  # columns 1, 4, 7, ...
    z_vals = hand[:, 2::3]  # columns 2, 5, 8, ...
    
    print(f"  {hand_name}:")
    print(f"    Wrist (frame 0): x={hand[0,0]:.4f}, y={hand[0,1]:.4f}, z={hand[0,2]:.4f}")
    print(f"    X range: [{x_vals.min():.4f}, {x_vals.max():.4f}] mean={x_vals.mean():.4f}")
    print(f"    Y range: [{y_vals.min():.4f}, {y_vals.max():.4f}] mean={y_vals.mean():.4f}")
    print(f"    Z range: [{z_vals.min():.4f}, {z_vals.max():.4f}] mean={z_vals.mean():.4f}")
    
    # Check for out-of-range x/y values (should be [0, 1])
    if x_vals.min() < -0.1 or x_vals.max() > 1.1:
        print(f"    ⚠️  X VALUES OUT OF [0,1] RANGE!")
    if y_vals.min() < -0.1 or y_vals.max() > 1.1:
        print(f"    ⚠️  Y VALUES OUT OF [0,1] RANGE!")


def compare_sources(python_data, flutter_data):
    """Compare Python and Flutter data side by side."""
    print("\n" + "=" * 70)
    print("PARITY COMPARISON")
    print("=" * 70)
    
    print(f"\n📊 Shapes:")
    print(f"  Python:  {python_data.shape}")
    print(f"  Flutter: {flutter_data.shape}")
    
    if python_data.shape != flutter_data.shape:
        print("\n❌ SHAPE MISMATCH — cannot compare element-wise.")
        return
    
    # Compare left hand wrist (first 3 values)
    print(f"\n🤚 Left Hand Wrist (frame 0):")
    py_lh = python_data[0, :3]
    fl_lh = flutter_data[0, :3]
    print(f"  Python:  x={py_lh[0]:.4f}, y={py_lh[1]:.4f}, z={py_lh[2]:.4f}")
    print(f"  Flutter: x={fl_lh[0]:.4f}, y={fl_lh[1]:.4f}, z={fl_lh[2]:.4f}")
    print(f"  Diff:    x={abs(py_lh[0]-fl_lh[0]):.4f}, y={abs(py_lh[1]-fl_lh[1]):.4f}, z={abs(py_lh[2]-fl_lh[2]):.4f}")
    
    # Compare right hand wrist (values 63-65)
    print(f"\n✋ Right Hand Wrist (frame 0):")
    py_rh = python_data[0, 63:66]
    fl_rh = flutter_data[0, 63:66]
    print(f"  Python:  x={py_rh[0]:.4f}, y={py_rh[1]:.4f}, z={py_rh[2]:.4f}")
    print(f"  Flutter: x={fl_rh[0]:.4f}, y={fl_rh[1]:.4f}, z={fl_rh[2]:.4f}")
    print(f"  Diff:    x={abs(py_rh[0]-fl_rh[0]):.4f}, y={abs(py_rh[1]-fl_rh[1]):.4f}, z={abs(py_rh[2]-fl_rh[2]):.4f}")
    
    # Handedness swap detection
    print(f"\n🔄 Handedness Swap Detection:")
    py_lh_present = np.any(python_data[:, :63] != 0)
    py_rh_present = np.any(python_data[:, 63:] != 0)
    fl_lh_present = np.any(flutter_data[:, :63] != 0)
    fl_rh_present = np.any(flutter_data[:, 63:] != 0)
    
    print(f"  Python:  LH={'✅' if py_lh_present else '❌'}  RH={'✅' if py_rh_present else '❌'}")
    print(f"  Flutter: LH={'✅' if fl_lh_present else '❌'}  RH={'✅' if fl_rh_present else '❌'}")
    
    if py_lh_present and not fl_lh_present and fl_rh_present and not py_rh_present:
        print("  ⚠️  POSSIBLE HANDEDNESS SWAP: Python has LH, Flutter has RH")
    elif py_rh_present and not fl_rh_present and fl_lh_present and not py_lh_present:
        print("  ⚠️  POSSIBLE HANDEDNESS SWAP: Python has RH, Flutter has LH")
    
    # X-axis mirroring detection
    print(f"\n🪞 X-Axis Mirroring Detection:")
    if py_lh_present and fl_lh_present:
        py_x_mean = python_data[:, 0::3][:, :21].mean()
        fl_x_mean = flutter_data[:, 0::3][:, :21].mean()
        mirror_test = abs(py_x_mean + fl_x_mean - 1.0)
        print(f"  LH X mean: Python={py_x_mean:.4f}, Flutter={fl_x_mean:.4f}")
        print(f"  Sum≈1.0 test: {mirror_test:.4f} {'(MIRRORED!)' if mirror_test < 0.15 else '(OK - not mirrored)'}")
    
    # Overall element-wise difference
    diff = np.abs(python_data - flutter_data)
    print(f"\n📏 Element-wise Difference:")
    print(f"  Mean absolute diff: {diff.mean():.6f}")
    print(f"  Max absolute diff:  {diff.max():.6f}")
    print(f"  Values within 0.01: {(diff < 0.01).mean()*100:.1f}%")
    print(f"  Values within 0.05: {(diff < 0.05).mean()*100:.1f}%")
    print(f"  Values within 0.10: {(diff < 0.10).mean()*100:.1f}%")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    python_file = os.path.join(script_dir, "debug_python_data.npy")
    flutter_file = os.path.join(script_dir, "debug_flutter_data.npy")
    
    python_data = load_debug_data(python_file)
    flutter_data = load_debug_data(flutter_file)
    
    if python_data is None and flutter_data is None:
        print("❌ No debug data files found!")
        print(f"   Expected: {python_file}")
        print(f"   Expected: {flutter_file}")
        print("\nTo generate:")
        print("  Python: Run realtime_inference_enhanced.py, press 'S' to save")
        print("  Flutter: Run the app and perform a sign (auto-saved by api_server)")
        sys.exit(1)
    
    # Analyze each source individually
    for name, data, filename in [("Python", python_data, python_file), 
                                  ("Flutter", flutter_data, flutter_file)]:
        if data is None:
            print(f"\n⚠️  {name} data not found: {filename}")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"{name.upper()} DATA ANALYSIS")
        print(f"{'=' * 70}")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        
        analyze_hand(data, "Left Hand", 0, 63)
        analyze_hand(data, "Right Hand", 63, 126)
    
    # Compare if both exist
    if python_data is not None and flutter_data is not None:
        compare_sources(python_data, flutter_data)
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
