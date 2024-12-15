import os
import glob
import cv2
import albumentations as A

# Base paths
DATASET_DIR = "Path/to/Dataset"  # Replace with your dataset directory path
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# Define advanced augmentation pipeline
augmentation_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.4),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])
)

def augment_images_and_labels(image_folder, label_folder):
    """Augment images and their corresponding YOLO labels."""
    augmented_image_folder = image_folder  # Augmented images go in the same folder
    augmented_label_folder = label_folder  # Augmented labels go in the same folder

    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
    print(f"Found {len(image_files)} images in {image_folder}")

    for image_file in image_files:
        print(f"Processing: {image_file}")
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(label_folder, f"{base_name}.txt")

        if not os.path.exists(label_file):
            print(f"Warning: Annotation file missing for {image_file}")
            continue

        # Load image and annotations
        image = cv2.imread(image_file)
        if image is None:
            print(f"Error: Unable to load {image_file}")
            continue

        h, w, _ = image.shape

        # Read YOLO labels
        with open(label_file, "r") as f:
            labels = []
            bboxes = []
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                bboxes.append([x_center, y_center, width, height])
                labels.append(int(class_id))

        # Apply augmentations
        augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=labels)

        # Generate unique filenames for augmented images and labels
        augmented_image_path = os.path.join(augmented_image_folder, f"{base_name}_aug.jpg")
        augmented_label_path = os.path.join(augmented_label_folder, f"{base_name}_aug.txt")

        # Save augmented image
        cv2.imwrite(augmented_image_path, augmented["image"])

        # Save augmented labels in YOLO format
        with open(augmented_label_path, "w") as f:
            for bbox, label in zip(augmented["bboxes"], augmented["class_labels"]):
                x_center, y_center, width, height = bbox
                f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"Saved: {augmented_image_path}, {augmented_label_path}")

def augment_dataset():
    """Apply augmentation to train and val sets."""
    print("Augmenting train set...")
    augment_images_and_labels(os.path.join(IMAGES_DIR, "train"), os.path.join(LABELS_DIR, "train"))

    print("Augmenting val set...")
    augment_images_and_labels(os.path.join(IMAGES_DIR, "val"), os.path.join(LABELS_DIR, "val"))

if __name__ == "__main__":
    augment_dataset()

