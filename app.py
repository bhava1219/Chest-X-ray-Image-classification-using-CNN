import streamlit as st
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load Models & Label Encoder
@st.cache_resource
def load_models():
    svm_model = joblib.load("svm_model.pkl")
    rf_model = joblib.load("random_forest_model.pkl")
    logreg_model = joblib.load("logistic_regression_model.pkl")
    knn_model = joblib.load("knn_model.pkl")
    dt_model = joblib.load("decision_tree_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return svm_model, rf_model, logreg_model, knn_model, dt_model, label_encoder

@st.cache_resource
def load_feature_extractor():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    return tf.keras.Model(inputs=base_model.input, outputs=base_model.output)

svm_model, rf_model, logreg_model, knn_model, dt_model, label_encoder = load_models()
feature_extractor = load_feature_extractor()

# Model Accuracies (update these to your actual results)
model_accuracies = {
    "SVM": 0.92,
    "Random Forest": 0.90,
    "Logistic Regression": 0.88,
    "KNN": 0.85,
    "Decision Tree": 0.83
}

st.title("Chest X-ray Classification (Pneumonia Detection)")
st.write("Upload a chest X-ray image (expected size for model input: 224×224 RGB)")

model_choice = st.selectbox("Choose Model", ["SVM", "Random Forest", "Logistic Regression", "KNN", "Decision Tree"])

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.session_state["prediction"] = None

    # Read image bytes and decode as color (3 channels)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize to 224x224 (MobileNetV2 input size)
    img_resized = cv2.resize(img, (224, 224))

    # Preprocess for MobileNetV2
    img_preprocessed = preprocess_input(img_resized.astype(np.float32))

    # Show uploaded image
    st.image(img_resized, caption="Uploaded X-ray (resized to 224×224)")

    # Extract features using MobileNetV2
    features = feature_extractor.predict(np.expand_dims(img_preprocessed, axis=0))
    features_flat = features.reshape(features.shape[0], -1)

    # Select model
    if model_choice == "SVM":
        model = svm_model
    elif model_choice == "Random Forest":
        model = rf_model
    elif model_choice == "Logistic Regression":
        model = logreg_model
    elif model_choice == "KNN":
        model = knn_model
    else:
        model = dt_model

    # Predict
    prediction = model.predict(features_flat)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]

    # Save prediction to session state
    st.session_state["prediction"] = predicted_label

# Display prediction
if "prediction" in st.session_state and st.session_state["prediction"]:
    st.subheader(f"Prediction: **{st.session_state['prediction']}**")
    st.write(f"Model Accuracy: {model_accuracies[model_choice] * 100:.2f}%")
