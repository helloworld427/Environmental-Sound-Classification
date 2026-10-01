"""
Noise Robustness Testing (Bonus Task)
-----------------------------------------
Systematically tests how model accuracy degrades as controlled noise
is added to the test set -- unlike the earlier ad-hoc WhatsApp/mic
tests, this uses precisely controlled noise levels (SNR), giving a
proper, reproducible robustness curve.

Uses the SAME 40 held-out test clips as the main evaluation, so results
are directly comparable to the clean-audio baseline (77.5% accuracy).
"""

import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"
CLASSES = ["dog_bark", "car_horn", "rain", "engine_sound", "human_speech"]
N_MFCC = 13
SAMPLE_RATE = 22050
RANDOM_STATE = 42

NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]


def add_noise(y, noise_std):
    if noise_std == 0.0:
        return y
    noise = np.random.normal(0, noise_std, y.shape)
    return y + noise


def extract_mfcc_from_audio(audio, sr, n_mfcc=N_MFCC):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)


def get_test_files():
    test_set = []
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        files = [f for f in os.listdir(class_dir) if f.lower().endswith('.wav')]
        _, test_files = train_test_split(files, test_size=0.2, random_state=RANDOM_STATE)
        for fname in test_files:
            test_set.append((os.path.join(class_dir, fname), class_name))
    return test_set


def run_robustness_test():
    print("Loading model...")
    model = joblib.load("sound_classifier_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    test_files = get_test_files()
    print(f"Testing on {len(test_files)} held-out clips across {len(NOISE_LEVELS)} noise levels\n")

    results = {}
    per_class_at_worst = {}

    for noise_std in NOISE_LEVELS:
        y_true = []
        y_pred = []

        for filepath, true_class in test_files:
            audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
            noisy_audio = add_noise(audio, noise_std)
            feat = extract_mfcc_from_audio(noisy_audio, sr).reshape(1, -1)

            pred_encoded = model.predict(feat)[0]
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]

            y_true.append(true_class)
            y_pred.append(pred_label)

        acc = accuracy_score(y_true, y_pred)
        results[noise_std] = acc
        print(f"Noise level {noise_std:.3f}: accuracy = {acc:.2%}")

        if noise_std == max(NOISE_LEVELS):
            for cls in CLASSES:
                cls_indices = [i for i, t in enumerate(y_true) if t == cls]
                cls_correct = sum(1 for i in cls_indices if y_pred[i] == cls)
                per_class_at_worst[cls] = cls_correct / len(cls_indices) if cls_indices else 0

    plt.figure(figsize=(8, 5))
    plt.plot(list(results.keys()), list(results.values()), 'o-', linewidth=2)
    plt.xlabel('Noise Level (Gaussian noise std dev)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs. Noise Level (Robustness Test)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('noise_robustness_curve.png', dpi=150)
    print("\nPlot saved as noise_robustness_curve.png")

    print(f"\nPer-class accuracy at highest noise level ({max(NOISE_LEVELS)}):")
    for cls, acc in per_class_at_worst.items():
        print(f"  {cls:15s}: {acc:.2%}")

    return results, per_class_at_worst


if __name__ == "__main__":
    run_robustness_test()