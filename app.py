import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import os

# PAGE CONFIG

st.set_page_config(
    page_title="st.mango",
    
    layout="centered"
)

st.markdown("""
<style>
    /* Dark Slate Background */
    .stApp {
        background-color: #1F2A37 !important;
    }

    /* Main Container Card */
    div[data-testid="stVerticalBlock"]:has(.upload-marker) {
        background-color: #2D3748 !important;
        border-radius: 20px !important;
        padding: 20px 15px !important;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.3) !important;
        border: 1px solid #374151 !important;
        margin-bottom: 20px;
    }

    .upload-marker {
        display: none !important;
        height: 0 !important;
    }

    /* Top Main Title */
    .top-header-title {
        font-weight: 800;
        font-size: 22px;
        color: #FACC15;
        margin-top: 5px;
        margin-bottom: 15px;
        text-align: center;
        letter-spacing: 0.5px;
        font-family: system-ui, -apple-system, sans-serif;
    }

    /* Outer Title (UPLOAD THE IMAGE) */
    .outer-title {
        font-weight: 800;
        font-size: 18px;
        color: #F59E0B;
        margin-bottom: 15px;
        text-align: center;
        letter-spacing: 0.5px;
        font-family: system-ui, -apple-system, sans-serif;
    }

    /* File Uploader Custom UI Box */
    div[data-testid="stFileUploader"] > label {
        display: none !important;
    }

    div[data-testid="stFileUploader"] section {
        background-color: transparent !important;
        border: 2px dashed #4B5563 !important;
        border-radius: 16px !important;
        padding: 20px 10px !important;
    }

    div[data-testid="stFileUploader"] section div {
        color: #9CA3AF !important;
    }

    /* Buttons Styling */
    div.stButton > button {
        border-radius: 10px !important;
        height: 44px !important;
        font-weight: bold !important;
        font-size: 15px !important;
        border: none !important;
        width: 100% !important;
    }

    /* Back Button Style */
    div.stButton > button[kind="secondary"] {
        background-color: #6B7280 !important;
        color: #FFFFFF !important;
    }

    /* Submit Button Style */
    div.stButton > button[kind="primary"] {
        background-color:#F59E0B !important;
        color: #FFFFFF !important;
    }

    /* Output Results Table Styling */
    .results-header {
        font-weight: 700;
        color: #FFFFFF;
        font-size: 18px;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .results-table {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        background-color: rgba(31, 41, 55, 0.8);
        border: 1px solid #374151;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }

    .table-col {
        flex: 1;
        text-align: left;
        padding: 0 8px;
        border-right: 1px solid #374151;
    }

    .table-col:last-child {
        border-right: none;
    }

    .col-title {
        color: #9CA3AF;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 4px;
    }

    .col-value {
        color: #F59E0B;
        font-size: 16px;
        font-weight: 700;
        word-break: break-word;
    }

    /* Mobile Screen Adjustments (Screen width <= 640px) */
    @media (max-width: 640px) {
        .top-header-title {
            font-size: 18px;
        }
        .outer-title {
            font-size: 16px;
        }
        .results-table {
            flex-direction: column;
            gap: 12px;
        }
        .table-col {
            border-right: none;
            border-bottom: 1px solid #374151;
            padding-bottom: 8px;
            padding-left: 0;
        }
        .table-col:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# SETTINGS & DATA

IMG_SIZE = (224, 224)

classes = [
    "Fully Ripe",
    "Overripe",
    "Perished",
    "Semiripe",
    "Unripe",
    "Notmatch"
]

recommendations = {
    "Unripe": "Allow further ripening.",
    "Semiripe": "Suitable for short-term storage.",
    "Fully Ripe": "Suitable for immediate sale or consumption.",
    "Overripe": "Consider processing soon.",
    "Perished": "Discard safely.",
    "Notmatch": "Try another one."
}

# LOAD MODELS

@st.cache_resource
def load_models():
    efficient_model = tf.keras.models.load_model("efficientnet_mango.keras")
    resnet_model = tf.keras.models.load_model("resnet_mango.keras")
    return efficient_model, resnet_model

try:
    efficient_model, resnet_model = load_models()
except Exception:
    st.error("couldn't load Model files.")

# PREDICTION LOGIC


def predict_mango(image_path, model):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image_batch = tf.expand_dims(image, axis=0)

    prediction = model.predict(image_batch, verbose=0)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]
    confidence = prediction[predicted_index] * 100

    return predicted_class, confidence


def compare_models(image_path):
    efficient_class, efficient_confidence = predict_mango(image_path, efficient_model)
    resnet_class, resnet_confidence = predict_mango(image_path, resnet_model)

    if efficient_class == resnet_class:
        final_class = efficient_class
        final_confidence = (efficient_confidence + resnet_confidence) / 2
    else:
        if efficient_confidence >= resnet_confidence:
            final_class = efficient_class
            final_confidence = efficient_confidence
        else:
            final_class = resnet_class
            final_confidence = resnet_confidence

    return final_class, final_confidence

# SESSION STATE MANAGEMENT

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "final_class" not in st.session_state:
    st.session_state.final_class = ""
if "final_confidence" not in st.session_state:
    st.session_state.final_confidence = 0.0
if "recommendation" not in st.session_state:
    st.session_state.recommendation = ""
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# MAIN UI


st.markdown('<div class="top-header-title">Mango Maturity Level Classification</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="upload-marker"></div>', unsafe_allow_html=True)
    st.markdown('<div class="outer-title">UPLOAD THE IMAGE</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_file is None:
        uploaded_file = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            key="mango_uploader"
        )

        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            st.rerun()
    else:
        uploaded_file = st.session_state.uploaded_file

    # Flexible Mobile Preview Image
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        st.image(image, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back", use_container_width=True, type="secondary"):
            st.session_state.prediction_done = False
            st.session_state.final_class = ""
            st.session_state.final_confidence = 0.0
            st.session_state.recommendation = ""
            st.session_state.uploaded_file = None
            if "mango_uploader" in st.session_state:
                del st.session_state["mango_uploader"]
            st.rerun()

    with col2:
        submit_clicked = st.button("Submit", type="primary", use_container_width=True)

# PREDICTION PROCESS


if submit_clicked:
    if st.session_state.uploaded_file is not None:
        image = Image.open(st.session_state.uploaded_file).convert("RGB")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image.save(temp_file.name, format="JPEG")
            image_path = temp_file.name

        final_class, final_confidence = compare_models(image_path)

        os.remove(image_path)

        st.session_state.final_class = final_class
        st.session_state.final_confidence = final_confidence
        st.session_state.recommendation = recommendations.get(final_class, "")
        st.session_state.prediction_done = True
        st.rerun()
    else:
        st.warning("Please upload an image first!")

# PREDICTION RESULTS


if st.session_state.prediction_done:
    st.markdown('<div class="results-header">Prediction Results</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="results-table">
        <div class="table-col">
            <div class="col-title">Maturity Level:</div>
            <div class="col-value">{st.session_state.final_class}</div>
        </div>
        <div class="table-col">
            <div class="col-title">Confidence Score:</div>
            <div class="col-value">{st.session_state.final_confidence:.2f}%</div>
        </div>
        <div class="table-col">
            <div class="col-title">Recommendation:</div>
            <div class="col-value">{st.session_state.recommendation}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)