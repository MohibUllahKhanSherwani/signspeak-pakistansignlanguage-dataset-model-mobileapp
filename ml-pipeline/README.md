# SignSpeak Data Collection Module

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**SignSpeak** is a comprehensive machine learning pipeline for Pakistan Sign Language (PSL) recognition. This repository contains tools for data collection, model training, and real-time inference using MediaPipe landmark detection and LSTM neural networks.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Model Comparison](#model-comparison)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

SignSpeak is part of a larger Final Year Project (FYP) at COMSATS University Islamabad, Abbottabad Campus. This repository specifically handles the machine learning component:

- **Data Collection**: GUI-based tool for recording PSL sign sequences
- **Feature Extraction**: MediaPipe holistic landmark detection (pose + hands)
- **Model Training**: LSTM-based deep learning with optional data augmentation
- **Inference**: Real-time sign recognition with performance metrics

**Note**: This is the ML data collection and training module. The complete SignSpeak system includes a Flutter mobile app and FastAPI backend (developed separately).

---

## ✨ Features

### Data Collection
- ✅ Modern GUI for efficient data recording
- ✅ Pause/resume functionality for long sessions
- ✅ Real-time landmark visualization
- ✅ Progress tracking across multiple signs
- ✅ Keyboard shortcuts for streamlined workflow

### Model Training
- ✅ Baseline training (standard approach)
- ✅ Advanced training with data augmentation (3-5x dataset expansion)
- ✅ Automated model comparison and evaluation
- ✅ Early stopping and learning rate scheduling
- ✅ Model checkpointing (saves best model)

### Real-Time Inference
- ✅ Webcam-based sign recognition
- ✅ Model selection (baseline vs augmented)
- ✅ Live accuracy tracking
- ✅ Performance metrics (FPS, confidence scores)

### Data Augmentation
- ✅ Time warping (speed variations)
- ✅ Horizontal flipping (left/right hand swapping)
- ✅ Spatial transformations (scaling, translation, rotation)
- ✅ Gaussian noise injection
- ✅ Temporal cropping

---

## 💻 System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, Ubuntu 20.04+, or macOS 10.15+
- **Python**: 3.9, 3.10, or 3.11 (3.11 recommended)
- **RAM**: 8 GB minimum
- **Storage**: 5 GB free space
- **Camera**: Webcam for data collection and inference

### Recommended Requirements
- **Python**: 3.11.9
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)

### Tested Configuration
```
OS: Windows 11
Python: 3.11.9
TensorFlow: 2.15.0
NumPy: 1.26.4
OpenCV: 4.9.0.80
MediaPipe: 0.10.9
```

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd SignSpeak-DataCollection
```

### Step 2: Create Virtual Environment

**Windows (PowerShell)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD)**:
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

> **Note**: If PowerShell gives an execution policy error:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; import cv2; import mediapipe as mp; print('✅ All dependencies installed successfully')"
```

Expected output:
```
✅ All dependencies installed successfully
```

---

## 📁 Project Structure

```
SignSpeak-DataCollection/
├── ml_pipeline_data_collection/    # Main ML workspace
│   ├── MP_Data/                    # Collected landmark sequences (gitignored)
│   ├── actions.txt                 # List of PSL signs to recognize
│   ├── actions_config.py           # Configuration parameters
│   ├── data_augmentation.py        # Augmentation algorithms
│   ├── collect_data_gui.py         # Enhanced data collection GUI
│   ├── train_model.py              # Baseline training script
│   ├── train_model_with_augmentation.py  # Advanced training with augmentation
│   ├── compare_models.py           # Automated model comparison
│   ├── realtime_inference.py       # Basic inference script
│   ├── realtime_inference_enhanced.py    # Enhanced inference with model selection
│   ├── action_model.h5             # Trained model (after training)
│   ├── label_encoder.pkl           # Label encoder (after training)
│   └── links_to_words.txt          # Reference links to PSL dictionary
├── SRS/                            # Software Requirements Specification
│   └── srs.txt
├── SDD/                            # Software Design Document
│   └── sdd.txt
├── venv/                           # Virtual environment (gitignored)
├── requirements.txt                # Pinned Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 🎬 Quick Start

### 1. Prepare Actions List

Create or edit `ml_pipeline_data_collection/actions.txt`:

```bash
cd ml_pipeline_data_collection
notepad actions.txt  # Windows
# OR
nano actions.txt     # Linux/macOS
```

Add PSL signs (one per line):
```
hello
thankyou
please
yes
no
```

### 2. Collect Data

```bash
python collect_data_gui.py
```

**GUI Controls**:
- Add/remove signs using buttons
- Select sign from dropdown
- Click "START COLLECTING" to begin
- Press **SPACE** to pause/resume
- Press **ESC** to stop

**Recommended**: Collect 50 sequences per sign for optimal results.

### 3. Train Model

**Option A: Baseline Model** (faster, no augmentation)
```bash
python train_model.py
```

**Option B: Augmented Model** (recommended, better accuracy)
```bash
python train_model_with_augmentation.py --augment
```

**Option C: Automated Comparison** (trains both and compares)
```bash
python compare_models.py
```

### 4. Test Model

```bash
# Test default model
python realtime_inference.py

# OR test with model selection
python realtime_inference_enhanced.py --augmented
```

---

## 🔬 Advanced Usage

### Data Augmentation

Augmentation effectively increases your dataset by 3-5x without additional data collection:

```bash
# 3x augmentation (recommended)
python train_model_with_augmentation.py --augment --augment-multiplier 3

# 5x augmentation (for very small datasets)
python train_model_with_augmentation.py --augment --augment-multiplier 5

# Custom epochs
python train_model_with_augmentation.py --augment --epochs 150
```

**Augmentation Techniques**:
- Time warping (0.8x-1.2x speed)
- Horizontal flipping (mirror + hand swapping)
- Spatial scaling (0.9x-1.1x)
- Spatial translation (±10%)
- Rotation (±15°)
- Gaussian noise (1% std)
- Temporal cropping (±10%)

### Configuration

Edit `ml_pipeline_data_collection/actions_config.py`:

```python
# Recording parameters
SEQUENCE_LENGTH = 30        # Frames per sequence
NUM_SEQUENCES = 50          # Sequences per sign
FRAME_WAIT_MS = 50          # Delay between frames (ms)

# Model parameters
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001

# Inference parameters
PREDICTION_THRESHOLD = 0.5  # Minimum confidence
```

### Model Architecture

LSTM-based sequential model:
```
Input: (30 frames, 225 features)
├── LSTM(64, return_sequences=True)
├── LSTM(128, return_sequences=True)
├── LSTM(64)
├── Dense(64)
├── Dense(32)
└── Dense(num_classes, softmax)

Total params: ~500K
```

**Features**: 225 values per frame
- Pose: 33 landmarks × 3 coords = 99
- Left hand: 21 landmarks × 3 coords = 63
- Right hand: 21 landmarks × 3 coords = 63

---

## 📊 Model Comparison

### Comparing Baseline vs Augmented

```bash
# Automated comparison (recommended)
python compare_models.py
```

**Output**:
```
MODEL COMPARISON REPORT
==================================================
Dataset:
  Baseline:  100 sequences
  Augmented: 300 sequences (3x)

Accuracy:
  Baseline  - Train: 95.5%, Test: 78.2%  (Gap: 17.3%)
  Augmented - Train: 93.8%, Test: 89.5%  (Gap: 4.3%)

🏆 RECOMMENDATION: USE AUGMENTED MODEL
   ✅ +11.3% better test accuracy
   ✅ Reduced overfitting by 13.0%
```

### Manual Testing

```bash
# Test baseline
python realtime_inference_enhanced.py --baseline
# Perform 20 predictions, mark as correct/wrong
# Note accuracy

# Test augmented
python realtime_inference_enhanced.py --augmented
# Perform same 20 predictions
# Compare accuracy
```

**Keyboard Controls During Testing**:
- **SPACE**: Mark prediction as correct ✅
- **X**: Mark prediction as wrong ❌
- **R**: Reset statistics
- **Q**: Quit

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
```bash
# Ensure virtual environment is activated
# Look for (venv) in command prompt

# Windows
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. Camera Not Detected
**Problem**: "Camera error!" or black screen

**Solution**:
- Close other apps using camera (Zoom, Teams, etc.)
- Check Windows camera permissions: Settings → Privacy → Camera
- Try different camera index in code: `cv2.VideoCapture(1)`

#### 3. TensorFlow Warnings
**Warning**: `oneDNN custom operations are on...`

**This is normal** - It's an informational message, not an error. To suppress:
```bash
set TF_ENABLE_ONEDNN_OPTS=0  # Windows CMD
$env:TF_ENABLE_ONEDNN_OPTS=0  # PowerShell
export TF_ENABLE_ONEDNN_OPTS=0  # Linux/macOS
```

#### 4. Low Accuracy
**Problem**: Validation accuracy < 80%

**Solutions**:
- Collect more data (aim for 50+ sequences per sign)
- Use data augmentation (`--augment`)
- Ensure consistent signing across sequences
- Check lighting conditions during data collection

#### 5. Out of Memory (OOM)
**Problem**: Training crashes with OOM error

**Solutions**:
- Reduce `BATCH_SIZE` in `actions_config.py` (try 8 or 4)
- Reduce `EPOCHS` (try 100 instead of 200)
- Close other applications
- Consider using GPU if available

### GPU Acceleration (Optional)

If you have an NVIDIA GPU:

```bash
# Uninstall CPU version

 tensorflow
pip uninstall tensorflow

# Install GPU version
pip install tensorflow[and-cuda]==2.15.0
```

Verify GPU:
```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

---

## 📚 Documentation

Additional guides available in `/brain/` artifacts:
- `project_summary.md` - Complete FYP project overview
- `gui_features_guide.md` - Detailed GUI documentation
- `augmentation_guide.md` - Data augmentation deep dive
- `testing_workflow.md` - Comprehensive testing guide
- `inference_comparison_guide.md` - Model comparison workflow

---

## 🤝 Contributing

This is an academic FYP project. For issues or suggestions:

1. Check existing issues
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Authors**:
- AbuZar Babar (CIIT/FA22-BSE-133/ATD)
- Mohib Ullah Khan Sherwani (CIIT/FA22-BSE-125/ATD)
- M. Abdullah Umar (CIIT/FA22-BSE-126/ATD)

**Supervisor**: Dr. Rab Nawaz Jadoon

---

## 🙏 Acknowledgments

- MediaPipe team for landmark detection library
- PSL Dictionary (psl.org.pk) for reference signs
- TensorFlow/Keras community

---

## 📞 Support

For technical issues specific to this repository:
- Check [Troubleshooting](#troubleshooting) section
- Review documentation in `/brain/` artifacts
- Create an issue with detailed logs

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Python**: 3.9+ (3.11 recommended)  
**Status**: Active Development
