# 🚀 Comprehensive Setup Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step-by-Step Installation](#step-by-step-installation)
3. [Firebase Configuration](#firebase-configuration)
4. [Environment Variables](#environment-variables)
5. [Running the Application](#running-the-application)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **OS:** macOS, Linux, or Windows
- **Python:** 3.8 or higher
- **RAM:** 4GB minimum (8GB recommended for TensorFlow)
- **Disk Space:** 2GB minimum

### Required Accounts
- **GitHub Account** (for cloning repository)
- **Firebase Account** (free tier available at https://firebase.google.com)

### Software to Install
- Python with pip
- Git
- A code editor (VS Code, PyCharm, etc.)

---

## Step-by-Step Installation

### 1. Clone the Repository

```bash
# Navigate to your workspace
cd ~/Desktop/GVSU/project/

# Clone the project
git clone https://github.com/KommineniYaswanth/Project.git
cd Project
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**This will install:**
- Flask 3.0.0 - Web framework
- TensorFlow 2.14.0 - Machine learning
- Keras 2.14.0 - Neural networks
- pyrebase4 - Firebase integration
- Flask-WTF - CSRF protection
- python-dotenv - Environment variable management
- Gunicorn - Production server
- And other dependencies

### 5. Create .env File

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your editor
nano .env    # or use VSCode, Sublime, etc.
```

---

## Firebase Configuration

### Part 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Click **"Create a project"**
3. Enter project name: `emotion-recognition`
4. Click through the setup steps
5. Wait for project creation to complete

### Part 2: Enable Authentication

1. In Firebase Console, go to **Authentication**
2. Click **"Get started"**
3. Select **"Email/Password"**
4. Toggle **"Enable"**
5. Click **"Save"**

### Part 3: Get Your Credentials

1. In Firebase Console, click the **gear icon** (Settings)
2. Go to **"Project settings"**
3. Find **"Your apps"** section
4. If no app exists, click **"Create app"** → **"Web"**
5. Copy the Firebase config object:

```javascript
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_AUTH_DOMAIN",
  projectId: "YOUR_PROJECT_ID",
  storageBucket: "YOUR_STORAGE_BUCKET",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID",
  measurementId: "YOUR_MEASUREMENT_ID",
  databaseURL: "YOUR_DATABASE_URL"
};
```

### Part 4: Update .env File

Edit your `.env` file and fill in these values:

```env
FIREBASE_API_KEY=YOUR_API_KEY
FIREBASE_AUTH_DOMAIN=YOUR_AUTH_DOMAIN
FIREBASE_PROJECT_ID=YOUR_PROJECT_ID
FIREBASE_STORAGE_BUCKET=YOUR_STORAGE_BUCKET
FIREBASE_MESSAGING_SENDER_ID=YOUR_MESSAGING_SENDER_ID
FIREBASE_APP_ID=YOUR_APP_ID
FIREBASE_MEASUREMENT_ID=YOUR_MEASUREMENT_ID
FIREBASE_DATABASE_URL=YOUR_DATABASE_URL
```

---

## Environment Variables

### Complete .env Configuration

```env
# Flask Configuration
FLASK_ENV=development              # Change to 'production' for deployment
FLASK_SECRET_KEY=your-random-key   # Generate a strong random key

# Firebase Configuration (get from Firebase Console)
FIREBASE_API_KEY=AIzaSyB...
FIREBASE_AUTH_DOMAIN=emotion-recognition-xxx.firebaseapp.com
FIREBASE_PROJECT_ID=emotion-recognition-xxx
FIREBASE_STORAGE_BUCKET=emotion-recognition-xxx.appspot.com
FIREBASE_MESSAGING_SENDER_ID=123456789
FIREBASE_APP_ID=1:123456789:web:abc...
FIREBASE_MEASUREMENT_ID=G-ABC...
FIREBASE_DATABASE_URL=https://emotion-recognition-xxx.firebaseio.com/

# Upload Configuration
MAX_UPLOAD_SIZE_MB=10
ALLOWED_EXTENSIONS=png,jpg,jpeg

# Model Configuration
MODEL_PATH=Emomodel.h5
```

### Generate Strong Secret Key

**Option 1: Using Python**
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

**Option 2: Using OpenSSL**
```bash
openssl rand -hex 32
```

Copy the output and paste into `FLASK_SECRET_KEY` in `.env`

### Security Notes
✅ `.env` file is in `.gitignore` - won't be committed  
✅ Keep credentials safe and never share  
✅ Rotate credentials periodically  
✅ Use different keys for development and production  

---

## Running the Application

### Development Mode

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Start the application
python main.py
```

The app will be available at: **http://localhost:5000**

### Production Mode

```bash
# Set environment to production
export FLASK_ENV=production

# Run with Gunicorn (4 workers)
gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

---

## Testing the Setup

### 1. Test Virtual Environment
```bash
which python  # macOS/Linux
which python  # Windows (should show venv path)
```

### 2. Test TensorFlow
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

### 3. Test Firebase
```bash
python -c "import pyrebase; print('Pyrebase installed successfully')"
```

### 4. Test Flask
```bash
python -c "from flask import Flask; print('Flask installed successfully')"
```

### 5. Test Model Loading
```bash
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('Emomodel.h5')
print('Model loaded successfully')
print(f'Model input shape: {model.input_shape}')
"
```

### 6. Test Application
```bash
python main.py
# Then open http://localhost:5000 in your browser
```

---

## Troubleshooting

### Issue: "Python command not found"
**Solution:**
```bash
# Install Python from https://python.org
# Then try:
python3 --version
```

### Issue: "ModuleNotFoundError: No module named 'flask'"
**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Model file 'Emomodel.h5' not found"
**Solution:**
```bash
# Ensure you're running from project root
cd /path/to/Project

# Check if file exists
ls -la Emomodel.h5

# If not present, download from the repository
```

### Issue: "Firebase credentials error: 'NoneType' object is not subscriptable"
**Solution:**
```bash
# Check .env file has all Firebase credentials
cat .env

# Ensure FIREBASE_API_KEY and other keys are filled
# Restart the application
```

### Issue: "Connection refused" / "Port 5000 already in use"
**Solution:**
```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or use a different port in main.py
app.run(debug=True, port=5001)
```

### Issue: "SSL Certificate error when downloading packages"
**Solution:**
```bash
# Upgrade pip
pip install --upgrade pip

# Try installing with --trusted-host
pip install -r requirements.txt --trusted-host pypi.python.org
```

### Issue: "Permission denied"
**Solution (macOS/Linux):**
```bash
chmod +x main.py
chmod -x venv/bin/*  # Make sure venv is executable
```

### Issue: "TensorFlow takes too long to load"
**Solution:**
```bash
# This is normal on first run - TensorFlow is loading model
# Subsequent runs will be faster
# For faster development, use CPU-only TensorFlow:
pip uninstall tensorflow
pip install tensorflow-cpu
```

---

## Next Steps

After successful setup:

1. **Create test account** in Firebase Console
2. **Test login/signup** at http://localhost:5000/signup
3. **Upload test images** and verify predictions
4. **Check logs** in terminal for any issues
5. **Read README.md** for feature documentation

---

## Performance Tips

- Use tensorflow-cpu if GPU not available: `pip install tensorflow-cpu`
- Close other applications to free RAM
- Use JPEG format for faster image processing
- Run on Linux/macOS for better performance

---

## Getting Help

1. Check **Troubleshooting** section above
2. Review **logs** in application console
3. Check **Firebase Documentation**: https://firebase.google.com/docs
4. Review **TensorFlow Documentation**: https://www.tensorflow.org/guide
5. Search **Stack Overflow** for specific errors

---

**Version:** 2.0  
**Last Updated:** March 29, 2025  
**Status:** ✅ Production Ready
