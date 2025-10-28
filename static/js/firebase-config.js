// Import Firebase SDK
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

// Firebase configuration
const firebaseConfig = {
    "apiKey": "AIzaSyBK0tfnwv3Kr9mGmk6zhoFMTQ6qzoyJCVg",
    "authDomain": "emotion-recognition-eee6b.firebaseapp.com",
    "projectId": "emotion-recognition-eee6b",
    "storageBucket": "emotion-recognition-eee6b.appspot.com",
    "messagingSenderId": "399035430804",
    "appId": "1:399035430804:web:22ed785143db9878b08181",
    "measurementId": "G-ZVR3GE7C2N",
    "databaseURL": "https://emotion-recognition-eee6b-default-rtdb.firebaseio.com/"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// Google Login Function
document.getElementById("google-login").addEventListener("click", () => {
    signInWithPopup(auth, provider)
        .then((result) => {
            const user = result.user;
            console.log("Logged in:", user);
            window.location.href = "/"; // Redirect to homepage
        })
        .catch((error) => {
            console.error("Login Error:", error.message);
        });
});

// Logout Function
document.getElementById("logout").addEventListener("click", () => {
    signOut(auth)
        .then(() => {
            console.log("User logged out");
            window.location.href = "/login"; // Redirect to login page
        })
        .catch((error) => {
            console.error("Logout Error:", error.message);
        });
});
