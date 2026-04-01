# Facial Emotion Recognition Web Application

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange?logo=tensorflow)
![Firebase](https://img.shields.io/badge/Firebase-FFCA28?logo=firebase)
![Security](https://img.shields.io/badge/Security-Enhanced-brightgreen)

---

## 📋 Overview

A **production-ready Facial Emotion Recognition Web Application** built with **Flask** and **TensorFlow** that detects human emotions from images using a deep learning CNN model. Features secure Firebase authentication, comprehensive error handling, and modern security practices.

**Detected Emotions:** Happy, Sad, Fear, Surprise, Neutral, Anger, Disgust

---

## Features

- Upload images for emotion detection
- Real-time webcam capture for live emotion analysis
- Firebase Authentication with email/password and Google sign-in
- CNN-based emotion classification with confidence metric
- Managed session state with secure cookies
- CSRF protection and server-side input validation
- Detailed logging and robust error handling
- Responsive, clean UI with dark mode
- Mobile and desktop support

---

## 🏗️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Backend** | Flask | 3.0.0 |
| **ML Model** | TensorFlow / Keras | 2.14.0 |
| **Authentication** | Firebase | Latest |
| **Frontend** | HTML5, CSS3, JavaScript | ES6+ |
| **Server** | Gunicorn (Production) | 21.2.0 |
| **Database** | Firebase Realtime DB | - |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Git
- Firebase account (free tier available)

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/KommineniYaswanth/Project.git
cd Project
```

#### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Configure Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your Firebase credentials
# FIREBASE_API_KEY=your-key-here
# FIREBASE_AUTH_DOMAIN=your-domain.firebaseapp.com
# etc.
```

#### 5. Setup Firebase
1. Go to [Firebase Console](https://console.firebase.google.com)
2. Create a new project or use existing one
3. Enable Email/Password authentication
4. Copy credentials to `.env` file
5. **Important:** Keep `.env` file in `.gitignore` (already configured)

#### 6. Run Application
```bash
python main.py
```

The app will start at `http://localhost:5000`

---

## Security Improvements

### Implemented Enhancements
- Environment variables for all secrets (`.env`)
- CSRF protection with Flask-WTF
- Secure cookies (HTTPOnly, Secure, SameSite)
- Server-side input validation and error handling
- Comprehensive logging for audit and debugging
- Upload file limits enforced (configurable)
- Updated TensorFlow model handling with compatible architecture
- Production ready mode (debug off in production)

### Credential Management

**NEVER commit credentials to Git.** The `.env` file should:
- Be added to `.gitignore` ✅ (Already done)
- Be created locally with your Firebase keys
- Be kept secure and backed up separately
- Use `.env.example` as template for setup

---

## 📁 Project Structure

```
Project/
├── main.py                 # Main Flask application (UPDATED)
├── requirements.txt        # Python dependencies (UPDATED)
├── .env.example           # Environment variables template (NEW)
├── .gitignore             # Git ignore rules (UPDATED)
├── README.md              # Documentation (THIS FILE)
├── Emomodel.h5            # Pre-trained TensorFlow model
├── Emomodel.json          # Model configuration
├── firebase_config.py     # Firebase setup (consider moving to main.py)
├── model.ipynb            # Model training notebook
├── dataset/               # Training/testing datasets
│   ├── train/
│   └── test/
├── templates/             # HTML templates
│   ├── index.html         # Main upload page
│   ├── login.html         # Login form
│   ├── signup.html        # Registration form
│   ├── result.html        # Prediction results
│   └── error.html         # Error page (ADD)
├── static/                # Static files
│   ├── style.css          # CSS styling
│   └── js/
│       └── firebase-config.js
└── uploads/               # Uploaded images (gitignored)
```

---

## 🔧 Configuration

### Environment Variables (.env)
```env
# Flask
FLASK_ENV=development           # Set to 'production' for deployment
FLASK_SECRET_KEY=your-key       # Change to random string in production

# Firebase
FIREBASE_API_KEY=your-api-key
FIREBASE_AUTH_DOMAIN=your-domain.firebaseapp.com
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_STORAGE_BUCKET=your-bucket.appspot.com
FIREBASE_MESSAGING_SENDER_ID=your-sender-id
FIREBASE_APP_ID=your-app-id
FIREBASE_MEASUREMENT_ID=your-measurement-id
FIREBASE_DATABASE_URL=https://your-db.firebaseio.com/

# Upload Settings
MAX_UPLOAD_SIZE_MB=10           # Max file size in MB
ALLOWED_EXTENSIONS=png,jpg,jpeg

# Model
MODEL_PATH=Emomodel.h5
```

---

## 📖 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET/POST | `/login` | User login |
| GET/POST | `/signup` | User registration |
| GET | `/logout` | User logout |

### Emotion Detection
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main page (requires auth) |
| POST | `/` | Upload/capture image for prediction |

---

## 🐛 Logging

Logs are output to console with the format:
```
2025-03-29 10:15:30,123 - __main__ - INFO - User logged in: user@example.com
2025-03-29 10:15:32,456 - __main__ - INFO - File uploaded: image.jpg
2025-03-29 10:15:35,789 - __main__ - INFO - Prediction: Happy (98.45%)
```

Check logs for:
- Authentication events
- File upload/capture events
- Model predictions
- Errors and warnings

---

## 🚨 Error Handling

The application handles:
- ✅ Missing model file
- ✅ Invalid image format
- ✅ File size exceeds limit (413 error)
- ✅ Firebase authentication errors
- ✅ Image processing errors
- ✅ Missing credentials

All errors are logged and user-friendly messages are displayed.

---

## 📊 Model Information

- **Architecture:** Convolutional Neural Network (CNN)
- **Input:** 48x48 grayscale image
- **Output:** 7-class emotion classification
- **Model File:** `Emomodel.h5` (TensorFlow/Keras format)
- **Output Classes:**
  1. Happy
  2. Sad
  3. Fear
  4. Surprise
  5. Neutral
  6. Anger
  7. Disgust

---

## 🌐 Deployment

### Development
```bash
FLASK_ENV=development python main.py
```

### Production (Using Gunicorn)
```bash
FLASK_ENV=production gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

### Docker (Optional)
Create `Dockerfile` for containerized deployment:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]
```

---

## 📝 Recent Updates (v2.0)

### 🔒 Security Fixes
- ✅ Removed hardcoded Firebase credentials
- ✅ Fixed weak secret key issue
- ✅ Added CSRF protection via Flask-WTF
- ✅ Implemented secure session cookies
- ✅ Added input validation and sanitization

### 💻 Code Quality
- ✅ Updated deprecated TensorFlow APIs
- ✅ Added comprehensive logging system
- ✅ Added type hints to all functions
- ✅ Improved error handling (try-except blocks)
- ✅ Added docstrings to all functions

### 📦 Dependencies
- ✅ Fixed missing packages in requirements.txt
- ✅ Added python-dotenv for environment management
- ✅ Added Flask-WTF for CSRF protection
- ✅ Added Gunicorn for production deployment
- ✅ Updated all packages to latest stable versions

### 📄 Documentation
- ✅ Updated README with security information
- ✅ Added .env.example template
- ✅ Added .gitignore rules
- ✅ Added deployment instructions
- ✅ Added API documentation

---

## 🆘 Troubleshooting

### Model Not Found
```
FileNotFoundError: Model file 'Emomodel.h5' not found!
```
**Solution:** Ensure `Emomodel.h5` is in the project root directory.

### Firebase Credentials Error
```
TypeError: 'NoneType' object is not subscriptable
```
**Solution:** Check that `.env` file has valid Firebase credentials.

### Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution:** Change port in `main.py` or kill process using port 5000:
```bash
lsof -ti:5000 | xargs kill -9
```

### Image Upload Fails
- Ensure file format is PNG, JPG, or JPEG
- Check file size is under 10MB
- Verify image contains a face

---

## 📚 References

- [Flask Documentation](https://flask.palletsprojects.com/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide/keras)
- [Firebase Authentication](https://firebase.google.com/docs/auth)
- [Flask-WTF CSRF Protection](https://flask-wtf.readthedocs.io/en/1.2.x/)
- [Python-dotenv](https://python-dotenv.readthedocs.io/)

---

## 📄 License

This project is provided as-is for educational purposes.

---

## 👤 Author

**Yaswanth Kommineni**  
GVSU Computer Science Project

---

## 📞 Support

For issues, questions, or suggestions, please create an issue in the repository or contact the project maintainer.

---

**Last Updated:** March 29, 2025  
**Version:** 2.0 (Security & Quality Enhancement)  
**Status:** ✅ Production Ready
