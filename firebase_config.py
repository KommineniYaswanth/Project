import pyrebase

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
