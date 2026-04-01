"""
Facial Emotion Recognition Web Application
Built with Flask, TensorFlow, and Firebase Authentication
"""

import base64
import logging
import os
from typing import Tuple

import cv2
from dotenv import load_dotenv
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import pyrebase

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__)

# Security Configuration
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-key-change-in-production")
app.config["SESSION_COOKIE_SECURE"] = os.getenv("FLASK_ENV") == "production"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Strict"

# CSRF Protection
csrf = CSRFProtect(app)

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Upload Configuration
UPLOAD_SUBFOLDER = "uploads"
UPLOAD_FOLDER = os.path.join(app.static_folder, UPLOAD_SUBFOLDER)
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", 16)) * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

# Firebase Configuration
FIREBASE_WEB_DEFAULTS = {
    "apiKey": "AIzaSyBK0tfnwv3Kr9mGmk6zhoFMTQ6qzoyJCVg",
    "authDomain": "emotion-recognition-eee6b.firebaseapp.com",
    "projectId": "emotion-recognition-eee6b",
    "storageBucket": "emotion-recognition-eee6b.appspot.com",
    "messagingSenderId": "399035430804",
    "appId": "1:399035430804:web:22ed785143db9878b08181",
    "measurementId": "G-ZVR3GE7C2N",
    "databaseURL": "https://emotion-recognition-eee6b-default-rtdb.firebaseio.com/"
}

raw_firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL")
}


def is_placeholder_value(value: str) -> bool:
    """Detect missing or placeholder Firebase values from `.env`."""
    if value is None:
        return True
    normalized = str(value).strip()
    return normalized == "" or normalized.startswith("your-") or "placeholder" in normalized.lower()


firebase_config = {
    key: (raw_firebase_config.get(key) if not is_placeholder_value(raw_firebase_config.get(key)) else FIREBASE_WEB_DEFAULTS.get(key, ""))
    for key in FIREBASE_WEB_DEFAULTS
}


def get_client_firebase_config() -> dict:
    """Return Firebase web config for browser-based authentication."""
    return dict(firebase_config)


def render_auth_template(template_name: str, **context):
    """Render auth templates with Firebase settings for Google sign-in."""
    client_config = get_client_firebase_config()
    google_login_enabled = bool(
        client_config.get("apiKey", "").startswith("AIza")
        and client_config.get("authDomain")
        and client_config.get("appId")
    )
    return render_template(
        template_name,
        firebase_web_config=client_config,
        google_login_enabled=google_login_enabled,
        **context,
    )

firebase = None
auth = None

try:
    if firebase_config.get("apiKey", "").startswith("AIza"):
        firebase = pyrebase.initialize_app(firebase_config)
        auth = firebase.auth()
        logger.info("Firebase initialized successfully")
    else:
        logger.warning("Firebase credentials are placeholders or invalid. Using development mode.")
        logger.warning("Update .env file with real Firebase credentials for production.")
except Exception as e:
    logger.error(f"Firebase initialization failed: {e}")
    logger.warning("Continuing in development mode without Firebase")

# Model Configuration
DEFAULT_MODEL_PATH = "Emomodel_improved.h5" if os.path.exists("Emomodel_improved.h5") else "Emomodel.h5"
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
MODEL_JSON_PATH = os.getenv("MODEL_JSON_PATH", "Emomodel.json")
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load Model with Error Handling (Lazy Loading)
model = None
face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_EMOTION_CONFIDENCE", 35))
MIN_MARGIN_THRESHOLD = float(os.getenv("MIN_EMOTION_MARGIN", 12))


def build_emotion_model() -> tf.keras.Model:
    """Rebuild the trained CNN architecture so compatible weights can be loaded."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 48, 1), name="input_layer"),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", name="conv2d"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d"),
        tf.keras.layers.Dropout(0.4, name="dropout"),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu", name="conv2d_1"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1"),
        tf.keras.layers.Dropout(0.4, name="dropout_1"),
        tf.keras.layers.Conv2D(512, (3, 3), activation="relu", name="conv2d_2"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_2"),
        tf.keras.layers.Dropout(0.4, name="dropout_2"),
        tf.keras.layers.Conv2D(512, (3, 3), activation="relu", name="conv2d_3"),
        tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_3"),
        tf.keras.layers.Dropout(0.4, name="dropout_3"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(512, activation="relu", name="dense"),
        tf.keras.layers.Dropout(0.4, name="dropout_4"),
        tf.keras.layers.Dense(256, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.3, name="dropout_5"),
        tf.keras.layers.Dense(7, activation="softmax", name="dense_2"),
    ])


def load_model() -> tf.keras.Model:
    """Load the pre-trained emotion recognition model."""
    global model
    try:
        if model is not None:
            return model

        if os.path.exists(MODEL_PATH):
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                logger.info(f"Model loaded successfully from {MODEL_PATH}")
                return model
            except Exception as direct_load_error:
                logger.warning(f"Direct model load failed: {str(direct_load_error)[:100]}")

            try:
                model = build_emotion_model()
                model.load_weights(MODEL_PATH)
                logger.info(f"Loaded trained weights from {MODEL_PATH} using compatible architecture")
                return model
            except Exception as weights_error:
                logger.warning(f"Compatible weight loading failed: {str(weights_error)[:100]}")

        logger.warning(f"Model files not usable at {MODEL_PATH} / {MODEL_JSON_PATH}, creating fallback model")
        return create_fallback_model()
    except Exception as e:
        logger.warning(f"Could not load model ({str(e)[:100]}), creating fallback model")
        return create_fallback_model()


def create_fallback_model() -> tf.keras.Model:
    """Create a simple fallback CNN model for emotion recognition."""
    try:
        fallback_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(48, 48, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(7, activation="softmax")
        ])
        logger.info("Fallback model created successfully")
        return fallback_model
    except Exception as e:
        logger.error(f"Failed to create fallback model: {e}")
        return None


def preprocess_image(img_path: str) -> np.ndarray:
    """Read an image, validate it, crop the main face, and convert it to the model input format."""
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Error: Unable to read image file")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reject blank, extremely dark, or extremely flat images early.
    if float(np.mean(gray)) < 8 or float(np.std(gray)) < 5:
        raise ValueError("Error: The uploaded image is too dark, blank, or unclear.")

    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        logger.info("No face detected with Haar cascade; using the full image region for prediction")
        face_region = gray
    else:
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        face_region = gray[y:y + h, x:x + w]

    # Improve contrast so non-happy emotions are easier to separate.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(face_region)

    resized = cv2.resize(enhanced, (48, 48))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=(0, -1))

# Try to load model at startup
load_model()


def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(img_path: str) -> Tuple[str, float]:
    """
    Predict emotion from an image file.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Tuple of (emotion_label, confidence_score)
    """
    global model
    
    try:
        # Load model if not already loaded
        if model is None:
            model = load_model()
        
        if model is None:
            logger.error("Model not available for prediction")
            return "Model not loaded", 0.0
        
        if not os.path.exists(img_path):
            logger.warning(f"Image file not found: {img_path}")
            return "Error: File not found", 0.0
        
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array, verbose=0)[0]
        top_indices = np.argsort(prediction)[::-1]
        predicted_class = int(top_indices[0])
        second_class = int(top_indices[1])
        confidence_score = float(prediction[predicted_class] * 100)
        confidence_margin = float((prediction[predicted_class] - prediction[second_class]) * 100)

        emotion = EMOTION_CLASSES[predicted_class]
        second_emotion = EMOTION_CLASSES[second_class]

        if confidence_score < MIN_CONFIDENCE_THRESHOLD or confidence_margin < MIN_MARGIN_THRESHOLD:
            logger.info(
                f"Low-confidence prediction: {emotion} ({confidence_score:.2f}%), "
                f"runner-up={second_emotion}, margin={confidence_margin:.2f}%"
            )
            return f"Uncertain ({emotion}/{second_emotion})", confidence_score

        logger.info(
            f"Prediction: {emotion} ({confidence_score:.2f}%), "
            f"runner-up={second_emotion}, margin={confidence_margin:.2f}%"
        )
        return emotion, confidence_score
    except ValueError as e:
        logger.warning(str(e))
        return str(e), 0.0
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Error: Error in prediction", 0.0


# ==================== Authentication Routes ====================

@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login."""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        
        if not email or not password:
            return render_auth_template("login.html", error="Email and password are required")
        
        try:
            # Development mode: bypass Firebase if not configured
            if auth is None:
                logger.warning(f"Development mode: Logging in user {email} without Firebase")
                session.clear()
                session["user"] = email
                session["user_name"] = email.split("@")[0]
                session["auth_provider"] = "password"
                session["dev_mode"] = True
                session.permanent = False
                return redirect(url_for("upload_file"))
            
            # Production mode: use Firebase
            auth.sign_in_with_email_and_password(email, password)
            session.clear()
            session["user"] = email
            session["user_name"] = email.split("@")[0]
            session["auth_provider"] = "password"
            session.permanent = False
            logger.info(f"User logged in: {email}")
            return redirect(url_for("upload_file"))
        except Exception as e:
            error_message = str(e)
            logger.warning(f"Login failed for {email}: {error_message}")
            return render_auth_template("login.html", error=f"Login failed: {error_message}")

    return render_auth_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Handle user registration."""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        
        # Validation
        if not email or not password:
            return render_auth_template("signup.html", error="Email and password are required")
        
        if password != confirm_password:
            return render_auth_template("signup.html", error="Passwords do not match")
        
        if len(password) < 6:
            return render_auth_template("signup.html", error="Password must be at least 6 characters")
        
        try:
            # Development mode: bypass Firebase if not configured
            if auth is None:
                logger.warning(f"Development mode: Registering user {email} without Firebase")
                return render_auth_template("signup.html", success="Signup successful! You can now login.")
            
            # Production mode: use Firebase
            auth.create_user_with_email_and_password(email, password)
            logger.info(f"New user registered: {email}")
            return redirect(url_for("login"))
        except Exception as e:
            logger.error(f"Signup failed for {email}: {str(e)}")
            return render_auth_template("signup.html", error=f"Signup failed: {str(e)}")

    return render_auth_template("signup.html")


@app.route("/google-login", methods=["POST"])
@csrf.exempt
def google_login():
    """Handle Google sign-in from the Firebase web client."""
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip()
    display_name = str(data.get("name", "")).strip()
    id_token = str(data.get("idToken", "")).strip()

    if not email:
        return jsonify({"success": False, "error": "Google account email was not received."}), 400

    try:
        logger.info("Google login payload: %s", data)

        # For development/demo purposes, trust the client-side authentication.
        # In production, verify id_token with Firebase Admin SDK.
        session.clear()
        session["user"] = email
        session["user_name"] = display_name or email.split("@")[0]
        session["auth_provider"] = "google"
        session.permanent = False
        logger.info(f"Google login successful: {email}")
        return jsonify({"success": True, "redirect_url": url_for("upload_file")})
    except Exception as e:
        logger.error(f"Google login failed for {email or 'unknown user'}: {e}")
        return jsonify({"success": False, "error": str(e) or "Google login failed. Please try again."}), 400


@app.route("/logout")
def logout():
    """Handle user logout."""
    user = session.pop("user", None)
    session.pop("user_name", None)
    session.pop("auth_provider", None)
    session.pop("dev_mode", None)
    if user:
        logger.info(f"User logged out: {user}")
    return redirect(url_for("login"))


# ==================== Upload & Prediction Routes ====================

@app.route("/", methods=["GET", "POST"])
def upload_file():
    """Handle image upload and emotion prediction."""
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            file = request.files.get("file")
            captured_image_data = request.form.get("captured_image")

            filename = None
            file_path = None

            # Handle file upload
            if file and file.filename:
                if not allowed_file(file.filename):
                    logger.warning(f"Unsupported file type: {file.filename}")
                    return render_template("index.html", error="Please upload a PNG, JPG, or JPEG image.")

                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)
                logger.info(f"File uploaded: {filename}")

            # Handle captured image
            elif captured_image_data:
                try:
                    if "," not in captured_image_data:
                        raise ValueError("Invalid captured image data")

                    image_data = base64.b64decode(captured_image_data.split(",", 1)[1])
                    filename = "captured_image.jpg"
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    with open(file_path, "wb") as f:
                        f.write(image_data)
                    logger.info("Image captured successfully")
                except Exception as e:
                    logger.error(f"Error processing captured image: {e}")
                    return render_template("index.html", error="Failed to process captured image")

            else:
                logger.warning("No valid image provided for prediction")
                return render_template("index.html", error="No valid image provided. Please upload or capture an image.")

            # Make prediction
            if file_path:
                result, confidence = predict_image(file_path)

                if result == "Model not loaded" or result.startswith("Error:"):
                    return render_template("index.html", error=result.replace("Error: ", ""))

                return render_template(
                    "result.html",
                    prediction=result,
                    confidence=f"{confidence:.2f}",
                    filename=filename
                )

        except Exception as e:
            logger.error(f"Error in upload_file: {e}")
            return render_template("index.html", error="An error occurred during processing")

    return render_template("index.html")


# ==================== Error Handlers ====================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    logger.warning("File upload exceeded max size")
    return render_template("index.html", error="File size exceeds maximum limit"), 413


@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server error."""
    logger.error(f"Internal server error: {error}")
    return render_template("error.html", error="Internal server error"), 500


@app.errorhandler(404)
def not_found(error):
    """Handle page not found."""
    return render_template("error.html", error="Page not found"), 404


# ==================== Application Entry Point ====================

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_ENV") == "development"
    port = int(os.getenv("PORT", 8000))  # Changed default from 5000 to 8000
    
    logger.info(f"Starting Flask app (debug={debug_mode})")
    
    try:
        app.run(debug=debug_mode, host="0.0.0.0", port=port, use_reloader=True)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.warning(f"Port {port} is in use, trying port {port + 1}")
            app.run(debug=debug_mode, host="0.0.0.0", port=port + 1, use_reloader=True)
        else:
            raise
