# Emotion Recognition Web App

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3-green?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange?logo=tensorflow)
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?logo=firebase)

---

## Overview
This project is a **Facial Emotion Recognition Web Application** built with **Flask** (Python) that allows users to upload or capture images and predicts human emotions using a trained convolutional neural network model.  
It also integrates **Firebase Authentication** to manage secure login/sign-up.  

The app detects basic human emotions (Happy, Sad, Angry, etc.) from face images and displays the results interactively on a web interface.

---

## Features
- 🖼️ Upload images to detect emotions  
- 📸 Real-time emotion detection via webcam  
- 🔐 Firebase Authentication (Email & Google login)  
- 🤖 CNN-based emotion classification using a pre-trained model  
- 📦 Includes templates and dataset for smooth functionality  

---

## Tech Stack
- **Backend:** Python Flask  
- **AI/ML Model:** TensorFlow / Keras (`Emomodel.h5`)  
- **Authentication:** Firebase Authentication  
- **Frontend:** HTML, CSS, JavaScript

---

## Quick Setup (Linux/Mac)

You can automate the setup with this Bash script:

```bash
#!/bin/bash

set -e

echo "Cloning repository..."
git clone https://github.com/KommineniYaswanth/Project.git
cd Project || { echo "Failed to enter Project directory"; exit 1; }

echo "Creating virtual environment..."
python3 -m venv venv
echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting the app..."
echo "Open your browser at http://127.0.0.1:5000"
python main.py
