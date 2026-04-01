import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getAuth, GoogleAuthProvider, signInWithPopup } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

const firebaseConfig = window.firebaseWebConfig || {};
const googleLoginUrl = window.googleLoginUrl || "/google-login";
const googleButton = document.getElementById("google-login");
const messageBox = document.getElementById("google-login-message");
const csrfTokenInput = document.querySelector('input[name="csrf_token"]');

function showMessage(message, type = "error") {
    if (!messageBox) {
        alert(message);
        return;
    }

    messageBox.textContent = message;
    messageBox.className = type;
    messageBox.style.display = "block";
}

if (googleButton) {
    const hasConfig = Boolean(
        firebaseConfig.apiKey &&
        firebaseConfig.authDomain &&
        firebaseConfig.appId &&
        String(firebaseConfig.apiKey).startsWith("AIza")
    );

    if (!hasConfig) {
        googleButton.disabled = true;
        googleButton.title = "Enable Firebase Google sign-in to use this option.";
        showMessage("Google login is not configured yet. Please update the Firebase settings.", "error");
    } else {
        const app = initializeApp(firebaseConfig);
        const auth = getAuth(app);
        const provider = new GoogleAuthProvider();
        provider.setCustomParameters({ prompt: "select_account" });

        googleButton.addEventListener("click", async () => {
            const originalText = googleButton.textContent;
            googleButton.disabled = true;
            googleButton.textContent = "⏳ Connecting to Google...";

            try {
                const result = await signInWithPopup(auth, provider);
                const user = result.user;
                const idToken = await user.getIdToken();

                const response = await fetch(googleLoginUrl, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-CSRFToken": csrfTokenInput ? csrfTokenInput.value : "",
                    },
                    body: JSON.stringify({
                        email: user.email || "",
                        name: user.displayName || "",
                        idToken,
                    }),
                });

                const data = await response.json();
                if (!response.ok || !data.success) {
                    throw new Error(data.error || "Google login failed. Please try again.");
                }

                showMessage("Google login successful. Redirecting...", "success");
                window.location.href = data.redirect_url || "/";
            } catch (error) {
                console.error("Google Login Error:", error);
                showMessage(error.message || "Google login failed. Please try again.", "error");
                googleButton.disabled = false;
                googleButton.textContent = originalText;
            }
        });
    }
}
