# 📱 Digit Recognition App

A **mobile application** built with Flutter that recognizes handwritten digits (0–9) from images using a custom deep learning model.

## 🚀 Overview

This digit recognition system combines mobile technology with machine learning to provide real-time digit classification. Users can capture or upload images of handwritten digits, and the app will instantly predict the digit using our custom-trained neural network.

## 🏗️ Architecture

The application consists of three main components:

### 📱 Flutter Mobile App
- Captures digit images via camera or gallery
- Provides intuitive user interface
- Sends images to backend for processing
- Displays prediction results in real-time

### 🧠 Deep Learning Model
- Custom-trained neural network for digit recognition
- Optimized for accuracy and speed
- Trained specifically for this project
- Supports digits 0-9 classification

### 🌐 FastAPI Backend
- RESTful API service
- Handles image processing and model inference
- Communicates between Flutter frontend and ML model
- Returns predictions with confidence scores

## 🎯 Project Goals

- ✅ Develop a lightweight and fast digit recognition system for mobile devices
- ✅ Expose the deep learning model via web API for real-time inference
- ✅ Seamlessly integrate Flutter and Python (FastAPI) technologies
- ✅ Provide accurate digit classification with minimal latency

## 🛠️ Tech Stack

### Frontend
- **Flutter** - Cross-platform mobile development
- **Dart** - Programming language

### Backend
- **FastAPI** - High-performance web framework
- **Python** - Backend programming language
- **Uvicorn** - ASGI server

### Machine Learning
- **TensorFlow/PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **PIL/OpenCV** - Image processing

## 📋 Prerequisites

### For Flutter App
- Flutter SDK (>=3.0.0)
- Dart SDK
- Android Studio / Xcode
- Android/iOS device or emulator

### For Backend
- Python 3.8+
- pip package manager
- Virtual environment (recommended)


---

⭐ **Star this repository if you found it helpful!**
