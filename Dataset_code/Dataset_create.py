import os
import random
import requests
import shutil
import cv2

# Configuration
DATA_PATH = "Path/to/images"
OUTPUT_DIR = "Dataset/save/path"
MODEL_API_URL = "https://detect.roboflow.com/ops-pallet-detection/1"
API_KEY = "Roboflow_API_Key"

# Create output directories
TRAIN_DIR = os.path.join(OUTPUT_DIR, "images/train")
VAL_DIR = os.path.join(OUTPUT_DIR, "images/val")
TEST_DIR = os.path.join(OUTPUT_DIR, "images/test")
ANNOTATION_TRAIN_DIR = os.path.join(OUTPUT_DIR, "labels/train")
ANNOTATION_VAL_DIR = os.path.join(OUTPUT_DIR, "labels/val")
ANNOTATION_TEST_DIR = os.path.join(OUTPUT_DIR, "labels/test")

for directory in [TRAIN_DIR, VAL_DIR, TEST_DIR, ANNOTATION_TRAIN_DIR, ANNOTATION_VAL_DIR, ANNOTATION_TEST_DIR]:
    os.makedirs(directory, exist_ok=True)

def generate_yolo_annotations():
    """Automatically annotate images using the Roboflow API and save YOLO format `.txt` files."""
    for file in os.listdir(DATA_PATH):
        if file.endswith(".jpg"):
            image_path = os.path.join(DATA_PATH, file)

            # Read the image for dimensions
            image = cv2.imread(image_path)
            h, w, _ = image.shape

            # Convert image to bytes
            _, img_encoded = cv2.imencode(".jpg", image)

            # Send image to Roboflow API
            response = requests.post(
                f"{MODEL_API_URL}?api_key={API_KEY}",
                files={"file": img_encoded.tobytes()}
            )

            if response.status_code == 200:
                detections = response.json().get("predictions", [])
                labels = []

                for detection in detections:
                    x, y, width, height = detection["x"], detection["y"], detection["width"], detection["height"]
                    class_id = 0  # Assuming single class 'pallet'

                    # Convert bounding box to YOLO format (normalized)
                    x_center = x / w
                    y_center = y / h
                    box_width = width / w
                    box_height = height / h

                    labels.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

                # Save labels in YOLO format
                label_file_path = os.path.join(DATA_PATH, file.replace(".jpg", ".txt"))
                with open(label_file_path, "w") as label_file:
                    for label in labels:
                        label_file.write(label + "\n")

                print(f"Annotations saved for {file}")
            else:
                print(f"Failed to process {file}: {response.text}")


def split_dataset(split_ratios=(0.7, 0.2, 0.1)):
    """Split dataset into train, val, and test while keeping images and their .txt files together."""
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".jpg")]
    random.shuffle(files)

    # Calculate split indices
    total_files = len(files)
    train_end = int(split_ratios[0] * total_files)
    val_end = train_end + int(split_ratios[1] * total_files)

    # Split dataset
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    splits = [("train", train_files), ("val", val_files), ("test", test_files)]
    for split_name, split_files in splits:
        image_dir = os.path.join(OUTPUT_DIR, f"images/{split_name}")
        label_dir = os.path.join(OUTPUT_DIR, f"labels/{split_name}")

        for file in split_files:
            # Copy image
            shutil.copy(os.path.join(DATA_PATH, file), os.path.join(image_dir, file))
            
            # Copy corresponding label
            label_file = file.replace(".jpg", ".txt")
            if os.path.exists(os.path.join(DATA_PATH, label_file)):
                shutil.copy(os.path.join(DATA_PATH, label_file), os.path.join(label_dir, label_file))

        print(f"{split_name.capitalize()} set created with {len(split_files)} images.")


if __name__ == "__main__":
    # Step 1: Generate YOLO annotations
    print("Generating YOLO annotations...")
    generate_yolo_annotations()

    # Step 2: Split dataset into train, val, and test
    print("Splitting dataset...")
    split_dataset()
    print("Dataset split completed.")

