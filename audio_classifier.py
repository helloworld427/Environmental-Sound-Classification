"""
Environmental Sound Classification Pipeline
---------------------------------------------
GreenWave TechLabs -- ML Internship Screening Task

Classifies environmental sounds (Dog Bark, Car Horn, Rain, Engine Sound,
Human Speech) using MFCC features + a Random Forest classifier.

WHY MFCC:
MFCCs (Mel-Frequency Cepstral Coefficients) compress an audio signal's
spectral content into a small set of coefficients that closely mirror
how the human ear perceives sound -- emphasizing frequency bands humans
are more sensitive to. This makes them a compact, well-established
feature representation for distinguishing sound classes with very
different spectral "shapes" (e.g., the tonal engine hum vs. the broadband
noise of rain vs. the sharp transient of a car horn), while keeping the
feature vector small enough to train quickly without needing a GPU.

WHY RANDOM FOREST (as the primary model):
- Trains in seconds/minutes on a laptop, no GPU required
- Handles the moderate-dimensional MFCC feature vectors well
- Naturally gives feature importance, useful for the report's
  "explain your decisions" requirement
- Easy to explain clearly in a demo -- important since the task states
  they are evaluating understanding and reasoning, not just accuracy

Dataset expected structure (after downloading ESC-50):
    dataset/
        dog_bark/*.wav
        car_horn/*.wav
        rain/*.wav
        engine_sound/*.wav
        human_speech/*.wav

See README.md for exact ESC-50 download + folder setup instructions.
"""
import librosa
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ============================================================
# Config
# ============================================================

DATASET_DIR = "dataset"
CLASSES = [
    "dog_bark",
    "car_horn",
    "rain",
    "engine_sound"
]
N_MFCC = 13          # standard choice: 13 coefficients capture most
                      # perceptually relevant spectral information
SAMPLE_RATE = 22050   # librosa's default; sufficient for these sound classes
RANDOM_STATE = 42     # for reproducibility -- important to state in the report


# ============================================================
# Step 1: Feature extraction
# ============================================================

def extract_mfcc_from_audio(audio, sample_rate, n_mfcc=N_MFCC):
    """
    Extracts MFCC features from an already-loaded audio array.
    Used by both the normal path and the augmentation path so
    feature extraction stays identical either way.
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)


def extract_mfcc_features(file_path, n_mfcc=N_MFCC, sr=SAMPLE_RATE):
    """
    Loads an audio file and extracts MFCC features, averaged over time
    to produce a single fixed-length feature vector per clip (handles
    clips of varying duration, which raw MFCC sequences would not).
    """
    import librosa
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return extract_mfcc_from_audio(audio, sample_rate, n_mfcc)


from scipy.signal import butter, lfilter


def simulate_codec_compression(y, sr):
    """
    Simulates speech-style codec compression.

    This function is only used for the human_speech class.
    """
    low = 300
    high = 3400
    nyquist = sr / 2

    b, a = butter(
        4,
        [low / nyquist, high / nyquist],
        btype="band"
    )

    filtered = lfilter(b, a, y)

    quantization_noise = np.random.normal(
        0,
        0.003,
        filtered.shape
    )

    compressed = filtered + quantization_noise

    return compressed


def augment_audio(y, sr, class_name=None):
    """
    Returns augmented versions of one audio clip.
    """
    import random

    augmented = []

    # Original audio
    augmented.append(y.copy())

    # Add Gaussian noise
    noise = np.random.normal(
        0,
        0.005,
        y.shape
    )

    noisy_audio = y + noise
    augmented.append(noisy_audio)

    # Time shift using zero padding
    maximum_shift = int(sr * 0.2)

    shift_amount = random.randint(
        -maximum_shift,
        maximum_shift
    )

    if shift_amount > 0:
        shifted_audio = np.concatenate(
            [
                np.zeros(shift_amount),
                y[:-shift_amount]
            ]
        )

    elif shift_amount < 0:
        shifted_audio = np.concatenate(
            [
                y[-shift_amount:],
                np.zeros(-shift_amount)
            ]
        )

    else:
        shifted_audio = y.copy()

    augmented.append(shifted_audio)

    # Pitch shift
    pitch_steps = random.choice([-2, 2])

    pitch_shifted_audio = librosa.effects.pitch_shift(
        y,
        sr=sr,
        n_steps=pitch_steps
    )

    augmented.append(pitch_shifted_audio)

    # Time stretch
    stretch_rate = random.choice([0.9, 1.1])

    stretched_audio = librosa.effects.time_stretch(
        y,
        rate=stretch_rate
    )

    if len(stretched_audio) > len(y):
        stretched_audio = stretched_audio[:len(y)]

    elif len(stretched_audio) < len(y):
        stretched_audio = np.pad(
            stretched_audio,
            (0, len(y) - len(stretched_audio))
        )

    augmented.append(stretched_audio)

    # Apply codec simulation only to human speech
    if class_name == "human_speech":
        codec_audio = simulate_codec_compression(
            y,
            sr
        )

        augmented.append(codec_audio)

    return augmented


def load_dataset(dataset_dir=DATASET_DIR, classes=CLASSES, use_augmentation=True, test_size=0.2):
    """
    Splits ORIGINAL files into train/test first, then augments only
    the training files. This prevents augmented versions of the same
    clip leaking across the train/test boundary.
    """
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    skipped = 0

    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"WARNING: folder not found: {class_dir} -- skipping this class")
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith('.wav')]
        print(f"Loading {len(files)} files for class '{class_name}'...")

        # split ORIGINAL filenames first
        train_files, test_files = train_test_split(
            files, test_size=test_size, random_state=RANDOM_STATE
        )

        for fname in train_files:
            fpath = os.path.join(class_dir, fname)
            try:
                audio, sample_rate = librosa.load(fpath, sr=SAMPLE_RATE)
                versions = augment_audio(audio, sample_rate, class_name) if use_augmentation else [audio]
                for version in versions:
                    feat = extract_mfcc_from_audio(version, sample_rate)
                    train_features.append(feat)
                    train_labels.append(class_name)
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
                skipped += 1

        for fname in test_files:
            fpath = os.path.join(class_dir, fname)
            try:
                # test set stays CLEAN - no augmentation
                feat = extract_mfcc_features(fpath)
                test_features.append(feat)
                test_labels.append(class_name)
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
                skipped += 1

    print(f"\nTotal train samples: {len(train_features)}, test samples: {len(test_features)} (skipped: {skipped})")
    return (
        np.array(train_features),
        np.array(train_labels),
        np.array(test_features),
        np.array(test_labels)
    )


# ============================================================
# Step 2: Train and evaluate
# ============================================================

def train_and_evaluate(X_train, y_train, X_test, y_test):
    label_encoder = LabelEncoder()
    # fit encoder on combined labels so it knows all classes,
    # even if a class happens to be missing from one split
    label_encoder.fit(np.concatenate([y_train, y_test]))

    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train_encoded)

    y_pred = model.predict(X_test)

    # ---- Metrics ----
    acc = accuracy_score(y_test_encoded, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_encoded, y_pred, average='weighted'
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Environmental Sound Classification')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("\nConfusion matrix saved as confusion_matrix.png")

    # ---- Feature importance ----
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances)
    plt.xlabel('MFCC Coefficient Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance (which MFCCs matter most)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    print("Feature importance plot saved as feature_importance.png")

    # ---- Save model ----
    joblib.dump(model, 'sound_classifier_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("\nModel saved as sound_classifier_model.pkl")

    return model, label_encoder, acc

# ============================================================
# Step 3: Predict on a new/unseen file
# ============================================================

def predict_single_file(file_path, model, label_encoder):
    feat = extract_mfcc_features(file_path).reshape(1, -1)
    pred_encoded = model.predict(feat)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    pred_proba = model.predict_proba(feat)[0]
    confidence = max(pred_proba)
    return pred_label, confidence


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Loading dataset and extracting MFCC features...")
    print("=" * 60)
    X_train, y_train, X_test, y_test = load_dataset()

    if len(X_train) == 0:
        print("\nERROR: No audio files found. Check your dataset/ folder structure")
        print("matches: dataset/<class_name>/*.wav -- see README.md")
    else:
        model, label_encoder, accuracy = train_and_evaluate(X_train, y_train, X_test, y_test)
        print(f"\n{'='*60}")
        print(f"Pipeline complete. Final test accuracy: {accuracy:.2%}")
        print(f"{'='*60}")