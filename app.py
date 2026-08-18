import streamlit as st
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
from keras.preprocessing import image

# PAGE CONFIG
st.set_page_config(
    page_title="Fruit Quality AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
.block-container { padding: 2rem; }
.hero {
    background: rgba(255,255,255,0.08);
    padding: 40px;
    border-radius: 24px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    text-align: center;
}
.card {
    background: rgba(255,255,255,0.10);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 15px 30px rgba(0,0,0,0.3);
}
.image-frame {
    border-radius: 20px;
    padding: 10px;
    background: linear-gradient(145deg, #00c6ff, #0072ff);
}
.result-fresh {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: #003300;
    font-size: 22px;
    font-weight: bold;
}
.result-rotten {
    background: linear-gradient(135deg, #ff512f, #dd2476);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    color: white;
    font-size: 22px;
    font-weight: bold;
}
.stButton>button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 30px;
    padding: 12px 30px;
    font-size: 16px;
    font-weight: 600;
    border: none;
}
.stButton>button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
@st.cache_resource
def load_trained_model():
    return load_model(r"C:\Users\alish\Downloads\FOOD_QUALITY_DETECTION\model_2_1_26_2.h5")

model = load_trained_model()

# LOAD HISTORY
# =========================================================
@st.cache_data
def load_history():
    with open(r"C:\Users\alish\Downloads\FOOD_QUALITY_DETECTION\history_combined.pkl", "rb") as f:
        return pickle.load(f)

history = load_history()

# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🍎 Fruit Quality Detection System</h1>
    <h3>AI-Powered Fresh vs Rotten Classification</h3>
    <p style="font-size:18px;">
        EfficientNet-based deep learning system for smart food inspection
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

# MAIN CONTENT
col1, col2 = st.columns([1, 1])

# LEFT COLUMN 

# State for camera trigger
if "show_camera" not in st.session_state:
    st.session_state["show_camera"] = False
if "camera_img" not in st.session_state:
    st.session_state["camera_img"] = None

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📷 Camera (keep steady, good lighting)")

    if st.session_state["show_camera"]:
        button_text = "Close Camera"
    else:
        button_text = "Open Camera"

    camera_toggle = st.button(button_text, key="camera_toggle")

    if camera_toggle:
        st.session_state["show_camera"] = not st.session_state["show_camera"]
        if not st.session_state["show_camera"]:
            st.session_state["camera_img"] = None

    # Show camera input only if camera is open
    if st.session_state["show_camera"]:
        camera_image = st.camera_input("Take photo")
        if camera_image:
            cam_img = Image.open(camera_image).convert("RGB")
            # st.image(cam_img, use_column_width=True)
            st.session_state["camera_img"] = cam_img

    # st.write("")

    # st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Supported formats: JPG, PNG, JPEG",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        # Uploading an image automatically closes the camera
        st.session_state["show_camera"] = False
        st.session_state["camera_img"] = None

        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_column_width=True)

# RIGHT COLUMN 
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Prediction")

    def predict_image(img_pil):
        # Resize
        img = img_pil.resize((224, 224))

        # Convert to array
        img = np.array(img)

        # FIX 1: mirror camera image
        img = np.fliplr(img)

        # FIX 2: reduce blur/noise
        img = cv2.GaussianBlur(img, (3, 3), 0)

        img_array = np.expand_dims(img, axis=0)
        img_array = preprocess_input(img_array)

        pred = model.predict(img_array)[0][0]

        if pred > 0.5:
            return "Rotten", pred
        else:
            return "Fresh", 1 - pred

    if (uploaded_file or "camera_img" in st.session_state) and st.button("Run Quality Analysis"):
        input_img = img if uploaded_file else st.session_state["camera_img"]

        label, confidence = predict_image(input_img)

        if confidence < 0.65:
            st.warning("⚠️ Low confidence. Retake image with better lighting & plain background.")

        if label == "Fresh":
            st.markdown(
                f"<div class='result-fresh'>✅ FRESH<br>Confidence: {confidence:.2%}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-rotten'>❌ ROTTEN<br>Confidence: {confidence:.2%}</div>",
                unsafe_allow_html=True
            )

        st.progress(float(confidence))

    st.markdown('</div>', unsafe_allow_html=True)

# PERFORMANCE
st.write("")
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Training Performance")


if st.checkbox("Show Accuracy & Loss"):
    h1 = history["h1"]
    h2 = history["h2"]

    acc = h1['accuracy'] + h2['accuracy']
    val_acc = h1['val_accuracy'] + h2['val_accuracy']
    loss = h1['loss'] + h2['loss']
    val_loss = h1['val_loss'] + h2['val_loss']

    epochs_phase1 = len(h1['accuracy'])
    epochs_phase2 = len(h2['accuracy'])
    total_epochs = epochs_phase1 + epochs_phase2
    epochs = range(1, total_epochs + 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

    ax[0].plot(epochs, acc, label="Train Acc", marker='o')
    ax[0].plot(epochs, val_acc, label="Val Acc", marker='o')
    ax[0].axvline(x=epochs_phase1, color='r', linestyle='--', label='Fine-tuning Start')
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(epochs, loss, label="Train Loss", marker='o')
    ax[1].plot(epochs, val_loss, label="Val Loss", marker='o')
    ax[1].axvline(x=epochs_phase1, color='r', linestyle='--', label='Fine-tuning Start')
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)

    st.subheader("Confusion Matrix & Classification Report")

    col1, col2 = st.columns(2)

    with col1:
        cm_img = Image.open("cm.png")
        st.image(cm_img, caption="Confusion Matrix", use_container_width=True)

    with col2:
        cr_img = Image.open("cr.png")
        st.image(cr_img, caption="Classification Report", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("""
<p style="text-align:center; opacity:0.7;">
© 2025 • AI Fruit Quality System
</p>
""", unsafe_allow_html=True)

