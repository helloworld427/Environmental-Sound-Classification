import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Environmental Sound Classifier", page_icon="🔊")

@st.cache_resource
def load_model():
    model = joblib.load("sound_classifier_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder

model, label_encoder = load_model()

SR = 22050
N_MFCC = 13

def extract_mfcc_features(file_path_or_buffer):
    audio, sample_rate = librosa.load(file_path_or_buffer, sr=SR)
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC
    )
    return np.mean(mfccs.T, axis=0)

st.title("🔊 Environmental Sound Classifier")

st.write(
    "Upload a short audio clip: dog bark, car horn, rain, "
    "or engine sound."
)

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Extracting features and predicting..."):
        feat = extract_mfcc_features(uploaded_file).reshape(1, -1)
        pred_encoded = model.predict(feat)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        pred_proba = model.predict_proba(feat)[0]

    display_label = pred_label

    st.success(f"**Predicted class: {display_label}**")
    st.subheader("Confidence per class")
    class_names = label_encoder.inverse_transform(model.classes_)
    fig, ax = plt.subplots()
    bars = ax.barh(class_names, pred_proba * 100, color="#4CAF50")
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 100)
    for bar, prob in zip(bars, pred_proba):
        ax.text(prob * 100 + 1, bar.get_y() + bar.get_height()/2,
                 f"{prob:.1%}", va='center')
    st.pyplot(fig)

    st.caption("Note: this model was trained on ESC-50 dataset clips. Real-world audio from other sources (e.g. voice messaging apps) may be compressed differently and can affect prediction accuracy.")
else:
    st.info("👆 Upload a .wav file to get started. Try your test clips: test_dog_bark.wav, test_car_horn.wav, test_rain.wav")