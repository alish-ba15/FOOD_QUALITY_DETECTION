from pathlib import Path
import pickle

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_2_1_26_2.h5"
HISTORY_PATH = BASE_DIR / "history_combined.pkl"

st.set_page_config(
    page_title="Fruit Quality AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container { padding: 2rem; }
    .hero, .card {
        background: rgba(255,255,255,0.08);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 15px 30px rgba(0,0,0,0.3);
    }
    .hero { text-align: center; padding: 40px; }
    .result-fresh, .result-rotten {
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    .result-fresh { background: linear-gradient(135deg, #00b09b, #96c93d); color: #003300; }
    .result-rotten { background: linear-gradient(135deg, #ff512f, #dd2476); color: white; }
    .stButton > button {
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 30px;
        padding: 12px 30px;
        font-weight: 600;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)


@st.cache_data
def load_history():
    if not HISTORY_PATH.is_file():
        return None
    with HISTORY_PATH.open("rb") as history_file:
        return pickle.load(history_file)


try:
    model = load_trained_model()
except (OSError, ValueError, ImportError) as error:
    st.error(f"Could not load the trained model at `{MODEL_PATH.name}`: {error}")
    st.info("Place the model file in the repository root and install dependencies from requirements.txt.")
    st.stop()

st.markdown(
    """
    <div class="hero">
        <h1>🍎 Fruit Quality Detection System</h1>
        <h3>AI-Powered Fresh vs Rotten Classification</h3>
        <p style="font-size:18px;">EfficientNet-based deep learning system for smart food inspection</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

left_column, right_column = st.columns(2)

with left_column:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📷 Capture or upload an image")
    camera_image = st.camera_input("Take a photo")
    uploaded_file = st.file_uploader(
        "Or choose an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
    )

    image_source = uploaded_file if uploaded_file is not None else camera_image
    input_image = None
    if image_source is not None:
        try:
            input_image = Image.open(image_source).convert("RGB")
            st.image(input_image, caption="Selected image", use_container_width=True)
        except (OSError, ValueError) as error:
            st.error(f"Could not read the selected image: {error}")
    st.markdown("</div>", unsafe_allow_html=True)


def predict_image(image_pil):
    image_array = np.asarray(image_pil.resize((224, 224)).convert("RGB"))
    # Keep the preprocessing used by the original app.
    image_array = np.fliplr(image_array)
    image_array = cv2.GaussianBlur(image_array, (3, 3), 0)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    prediction = np.asarray(model.predict(image_array, verbose=0)).reshape(-1)
    if prediction.size == 0 or not np.isfinite(prediction[0]):
        raise ValueError("The model returned an invalid prediction.")
    rotten_score = float(prediction[0])
    if not 0.0 <= rotten_score <= 1.0:
        raise ValueError("Expected the model to return a probability between 0 and 1.")

    if rotten_score > 0.5:
        return "Rotten", rotten_score
    return "Fresh", 1.0 - rotten_score


with right_column:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Prediction")
    analyze = st.button("Run Quality Analysis", disabled=input_image is None)
    if analyze and input_image is not None:
        try:
            label, confidence = predict_image(input_image)
            if confidence < 0.65:
                st.warning("⚠️ Low confidence. Try a clearer image with good lighting and a plain background.")

            result_class = "result-fresh" if label == "Fresh" else "result-rotten"
            icon = "✅" if label == "Fresh" else "❌"
            st.markdown(
                f"<div class='{result_class}'>{icon} {label.upper()}<br>Confidence: {confidence:.2%}</div>",
                unsafe_allow_html=True,
            )
            st.progress(confidence)
            st.caption("Prediction confidence is not a food-safety guarantee.")
        except (ValueError, RuntimeError, OSError) as error:
            st.error(f"Could not analyze this image: {error}")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Training Performance")
history = load_history()
if history is None:
    st.info("Training history file not found; performance charts are unavailable.")
elif st.checkbox("Show Accuracy & Loss"):
    try:
        phase_one = history["h1"]
        phase_two = history["h2"]
        accuracy = list(phase_one["accuracy"]) + list(phase_two["accuracy"])
        validation_accuracy = list(phase_one["val_accuracy"]) + list(phase_two["val_accuracy"])
        loss = list(phase_one["loss"]) + list(phase_two["loss"])
        validation_loss = list(phase_one["val_loss"]) + list(phase_two["val_loss"])

        import matplotlib.pyplot as plt

        epochs = range(1, len(accuracy) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        axes[0].plot(epochs, accuracy, label="Train accuracy", marker="o")
        axes[0].plot(epochs, validation_accuracy, label="Validation accuracy", marker="o")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs, loss, label="Train loss", marker="o")
        axes[1].plot(epochs, validation_loss, label="Validation loss", marker="o")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except (KeyError, TypeError, ValueError) as error:
        st.warning(f"Training history has an unexpected format; charts are unavailable: {error}")

    matrix_path = BASE_DIR / "cm.png"
    report_path = BASE_DIR / "cr.png"
    if matrix_path.is_file() or report_path.is_file():
        st.subheader("Confusion Matrix & Classification Report")
        chart_columns = st.columns(2)
        for column, asset_path, caption in zip(
            chart_columns,
            (matrix_path, report_path),
            ("Confusion Matrix", "Classification Report"),
        ):
            if asset_path.is_file():
                with column:
                    st.image(str(asset_path), caption=caption, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; opacity:0.7;">© 2025 • AI Fruit Quality System</p>',
    unsafe_allow_html=True,
)
