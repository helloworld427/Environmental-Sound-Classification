import os
import shutil
import csv

ESC50_AUDIO_DIR = "ESC-50/audio"
ESC50_META_CSV = "ESC-50/meta/esc50.csv"
OUTPUT_DIR = "dataset"

CLASS_MAPPING = {
    "dog_bark": "dog",
    "car_horn": "car_horn",
    "rain": "rain",
    "engine_sound": "engine",
    "human_speech": "laughing",
}

def sort_dataset():
    for class_name in CLASS_MAPPING:
        os.makedirs(os.path.join(OUTPUT_DIR, class_name), exist_ok=True)

    counts = {class_name: 0 for class_name in CLASS_MAPPING}

    with open(ESC50_META_CSV, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row['filename']
            category = row['category']
            for our_class, esc50_category in CLASS_MAPPING.items():
                if category == esc50_category:
                    src = os.path.join(ESC50_AUDIO_DIR, filename)
                    dst = os.path.join(OUTPUT_DIR, our_class, filename)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                        counts[our_class] += 1
                    break

    print("Sorting complete. Files per class:")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count} files")

if __name__ == "__main__":
    sort_dataset()