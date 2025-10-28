import base64
from flask import Flask, request, render_template, redirect, url_for, session
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import io
from werkzeug.utils import secure_filename
import pyrebase
from firebase_config import auth

app = Flask(__name__)
app.secret_key = "supersecretkey" 

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Firebase Configuration
firebase_config = {
    "apiKey": "AIzaSyBK0tfnwv3Kr9mGmk6zhoFMTQ6qzoyJCVg",
    "authDomain": "emotion-recognition-eee6b.firebaseapp.com",
    "projectId": "emotion-recognition-eee6b",
    "storageBucket": "emotion-recognition-eee6b.appspot.com",
    "messagingSenderId": "399035430804",
    "appId": "1:399035430804:web:22ed785143db9878b08181",
    "measurementId": "G-ZVR3GE7C2N",
    "databaseURL": "https://emotion-recognition-eee6b-default-rtdb.firebaseio.com/"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Load the trained model
MODEL_PATH = os.path.join(os.getcwd(), "Emomodel.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file 'Emomodel.h5' not found!")

model = tf.keras.models.load_model(MODEL_PATH)

# Emotion labels
emotion_classes = ["Happy", "Sad", "Fear", "Surprise", "Neutral", "Anger", "Disgust"]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    try:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(48, 48), color_mode="grayscale")
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence_score = np.max(prediction) * 100

        return emotion_classes[predicted_class], confidence_score
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error in prediction", 0

# ---------------- Authentication Routes ----------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session["user"] = email
            return redirect(url_for("upload_file"))
        except Exception as e:
            error_message = str(e)  # Get the error message from Firebase
            return render_template("login.html", error=f"Login failed: {error_message}")

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            auth.create_user_with_email_and_password(email, password)
            return redirect(url_for("login"))
        except Exception as e:
            return render_template("signup.html", error=f"Signup failed: {str(e)}")

    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ---------------- Upload & Capture Route ----------------

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("file")
        captured_image_data = request.form.get("captured_image")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
        elif captured_image_data:
            image_data = base64.b64decode(captured_image_data.split(",")[1])
            filename = "captured_image.png"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(file_path, "wb") as f:
                f.write(image_data)
        else:
            return "No image provided", 400

        result, confidence = predict_image(file_path)
        return render_template("result.html", prediction=result, confidence=confidence, filename=filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
