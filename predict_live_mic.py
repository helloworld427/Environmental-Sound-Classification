"""
Real-Time Microphone Prediction (Bonus Task)
-----------------------------------------------
Listens to the live microphone, records a short clip, extracts MFCC
features the same way as training, and predicts the sound class in
real time -- using the same trained model, no retraining needed.

Install: pip install sounddevice soundfile
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import joblib
import time

MODEL_PATH = "sound_classifier_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
SAMPLE_RATE = 22050
N_MFCC = 13
RECORD_SECONDS = 3
TEMP_FILE = "_mic_capture.wav"


def record_audio(duration=RECORD_SECONDS, sr=SAMPLE_RATE):
    print(f"\nRecording for {duration} seconds... speak/make a sound now!")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    sf.write(TEMP_FILE, recording, sr)
    print("Recording complete.")
    return TEMP_FILE


def extract_mfcc_features(file_path, n_mfcc=N_MFCC, sr=SAMPLE_RATE):
    audio, sample_rate = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)


def predict_live(model, label_encoder):
    audio_file = record_audio()
    feat = extract_mfcc_features(audio_file).reshape(1, -1)

    pred_encoded = model.predict(feat)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    pred_proba = model.predict_proba(feat)[0]

    print(f"\nPredicted: {pred_label}")
    print("Confidence per class:")
    for cls, prob in zip(label_encoder.classes_, pred_proba):
        print(f"  {cls:15s}: {prob:.2%}")

    return pred_label, pred_proba


if __name__ == "__main__":
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("Model loaded. Ready for real-time prediction.")

    while True:
        input("\nPress Enter to record and predict (or Ctrl+C to quit)...")
        predict_live(model, label_encoder)