# SignSpeak: Word-Level Pakistan Sign Language Recognition System

![SignSpeak Banner](https://img.shields.io/badge/SignSpeak-PSL%20Recognition-blue?style=for-the-badge)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

**SignSpeak** is a comprehensive Final Year Project (FYP) designed to bridge the communication gap between the Deaf/Hard-of-Hearing community and hearing individuals in Pakistan. It is a mobile-based system that translates Pakistan Sign Language (PSL) gestures into text.

---

## 📖 Table of Contents
- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Documentation](#-documentation)
- [Authors](#-authors)

---

## 🎯 Overview

SignSpeak aims to provide an accessible and efficient communication tool for PSL users. The system leverages computer vision and machine learning to interpret sign language gestures captured via a smartphone camera and translates them into understandable text.

**Core Objectives:**
*   Real-time translation of PSL words.
*   Mobile-first approach for accessibility.
*   Lightweight and efficient architecture suitable for mid-range smartphones.

---

## 🏗 System Architecture

The SignSpeak ecosystem consists of three main components:

1.  **Mobile Application (Frontend)**: A Flutter-based app that serves as the user interface for capturing video frames and displaying translations.
2.  **Backend Server**: A FastAPI (Python) server that handles heavy computations, including landmark detection (MediaPipe) and ML inference (TensorFlow/Keras).
3.  **Admin Dashboard**: A web-based panel for managing user accounts, reviewing complaints, and monitoring system performance.

### High-Level Workflow
1.  **Capture**: User points camera at the signer via the Mobile App.
2.  **Process**: Frames are sent to the Backend Server.
3.  **Analyze**: Backend extracts landmarks using **MediaPipe** and feeds them into an **LSTM-based Neural Network**.
4.  **Translate**: Predicted word is returned to the app and displayed to the user.

---

## ✨ Key Features

*   **Real-time PSL Translation**: Instant recognition of sign language gestures.
*   **User Authentication**: Secure signup and login via Firebase.
*   **PSL Dictionary**: Built-in reference for learning implementation of standard signs.
*   **Complaint Management**: Users can report incorrect translations or bugs; Admins can review and resolve them.
*   **Profile Management**: extensive user profile customization.
*   **Accessibility Settings**: Adjustable font sizes and UI elements.

---

## 🛠 Technology Stack

| Component | Technology |
|-----------|------------|
| **Mobile App** | Flutter (Dart) |
| **Backend API** | FastAPI (Python) |
| **ML/AI** | TensorFlow, Keras, MediaPipe |
| **Database/Auth**| Firebase (Authentication, Firestore) |
| **Admin Web** | React/Next.js (Planned) |
| **DevOps** | Docker (Planned) |

---

## 📂 Repository Structure

```
d:/SignSpeak/SignSpeak-FYP/
├── ml-pipeline/          # Machine Learning Data Collection & Training Module
│   ├── SDD/              # Software Design Documents
│   ├── SRS/              # Software Requirements Specifications
│   ├── ml_pipeline_data_collection/ # Core ML scripts (Capture, Train, Inference)
│   └── ...
├── front-end-mobile/     # Flutter Mobile Application (Under Development)
├── front-end-web/        # Admin Dashboard Web App (Under Development)
└── README.md             # Master Project Documentation
```

---

## 🚀 Getting Started

Currently, the **ML Pipeline** is the active component available for setup and testing.

### Prerequisites
*   Python 3.9 - 3.11
*   Webcam

### Running the ML Pipeline
Please refer to the detailed [ML Pipeline README](ml-pipeline/README.md) for instructions on how to:
1.  Install dependencies.
2.  Collect your own dataset using the GUI.
3.  Train the LSTM model.
4.  Run real-time inference via webcam.

---

## 📚 Documentation

Detailed documentation can be found within the `ml-pipeline` directory:
*   [**Software Requirements Specification (SRS)**](ml-pipeline/SRS/srs.txt): Detailed functional and non-functional requirements.
*   [**Software Design Document (SDD)**](ml-pipeline/SDD/sdd.txt): Architecture, data flow, and component design.

---

## 👥 Authors

**COMSATS University Islamabad, Abbottabad Campus**
*   **AbuZar Babar** (CIIT/FA22-BSE-133/ATD)
*   **Mohib Ullah Khan Sherwani** (CIIT/FA22-BSE-125/ATD)
*   **M. Abdullah Umar** (CIIT/FA22-BSE-126/ATD)

**Supervisor:**
*   Dr. Rab Nawaz Jadoon

---

