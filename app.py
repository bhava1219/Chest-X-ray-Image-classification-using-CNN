import streamlit as st
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------
# Load CNN Model
# ---------------------------
@st.cache_resource
def load_cnn_model(model_path="cnn_pneumonia_model.h5"):
    cnn_model = load_model(model_path)
    return cnn_model

cnn_model = load_cnn_model()

# CNN classes
categories = ["Normal", "Pneumonia"]

# ---------------------------
# Load Traditional ML Models
# ---------------------------
@st.cache_resource
def load_traditional_models():
    svm_model = joblib.load("svm_model.pkl")
    rf_model = joblib.load("random_forest_model.pkl")
    dt_model = joblib.load("decision_tree_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    # Preload feature extractor to save time
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    return svm_model, rf_model, dt_model, label_encoder, feature_extractor

svm_model, rf_model, dt_model, label_encoder, feature_extractor = load_traditional_models()

# Model accuracies
model_accuracies = {
    "SVM": 0.96,
    "Random Forest": 0.95,
    "Decision Tree": 0.86,
    "CNN": 0.89
}

st.title("Chest X-ray Classification (Pneumonia Detection)")
st.write("Upload a chest X-ray image")

# ---------------------------
# Dropdown always visible
# ---------------------------
model_choice = st.selectbox("Choose Model", ["CNN", "SVM", "Random Forest", "Decision Tree"])

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.session_state["prediction"] = None

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, caption="Uploaded X-ray")

    if model_choice == "CNN":
        # Preprocess for CNN
        img_resized = cv2.resize(img, (150, 150))
        img_input = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)

        # Predict
        pred_probs = cnn_model.predict(img_input)
        pred_class = int(np.argmax(pred_probs))
        predicted_label = categories[pred_class]

    else:
        # Traditional models: extract features
        img_resized = cv2.resize(img, (224, 224))
        img_preprocessed = preprocess_input(img_resized.astype(np.float32))
        features = feature_extractor.predict(np.expand_dims(img_preprocessed, axis=0))
        features_flat = features.reshape(features.shape[0], -1)

        if model_choice == "SVM":
            model = svm_model
        elif model_choice == "Random Forest":
            model = rf_model
        else:
            model = dt_model

        prediction = model.predict(features_flat)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

    st.session_state["prediction"] = predicted_label

# Display prediction
if "prediction" in st.session_state and st.session_state["prediction"]:
    st.subheader(f"Prediction: **{st.session_state['prediction']}**")
    st.write(f"Model Accuracy: {model_accuracies[model_choice] * 100:.2f}%")
