import librosa
import numpy as np
import joblib

model = joblib.load("sound_classifier_model.pkl")
SR = 22050
N_MFCC = 13

# Map numeric labels back to class names - alphabetical order (adjust if wrong)
class_names = {
    0: "car_horn",
    1: "dog_bark",
    2: "engine_sound",
    3: "human_speech",
    4: "rain"
}

def predict_single_file(filepath):
    y, sr = librosa.load(filepath, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feat = np.mean(mfcc.T, axis=0).reshape(1, -1)

    prediction = model.predict(feat)[0]
    probabilities = model.predict_proba(feat)[0]

    predicted_label = class_names.get(prediction, str(prediction))

    print(f"\nFile: {filepath}")
    print(f"Predicted class: {predicted_label}")
    print("Confidence per class:")
    for cls, prob in zip(model.classes_, probabilities):
        cls_label = class_names.get(cls, str(cls))
        print(f"  {cls_label:15s}: {prob:.2%}")

if __name__ == "__main__":
    test_files = [
        "test_dog_bark.wav",
        "test_car_horn.wav",
        "test_rain.wav"
    ]
    for f in test_files:
        predict_single_file(f)
