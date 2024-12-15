import os
from ultralytics import YOLO

DATASET_DIR = "dataset/path"  # Dataset path
MODEL_SAVE_DIR = "Model/path"  # Path to save model and results
BATCH_SIZE = 8
EPOCHS = 75  # Adjust as needed
LEARNING_RATE = 0.001

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Verify dataset structure
print("Contents of the dataset directory:")
!ls $DATASET_DIR
print("\nTrain folder contents:")
!ls $DATASET_DIR/images/train
print("\nValidation folder contents:")
!ls $DATASET_DIR/images/val

# Ensure the dataset has the correct folder structure
required_dirs = ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]
for folder in required_dirs:
    if not os.path.exists(os.path.join(DATASET_DIR, folder)):
        raise ValueError(f"Missing '{folder}' directory in the dataset! Ensure your dataset is structured correctly.")


def train_yolo():
    """Train YOLOv11 model for pallet detection."""
    # Load the pre-trained YOLOv11 model (ensure the model file is available or download it)
    model = YOLO("yolo11n.pt")  # Load a pre-trained YOLOv11 small model

    # Train the model on the dataset
    results = model.train(
        data="/home/varchasvin/Peer_Robotics/src/Detection/Pallets.yaml",  # Path to YAML file
        epochs=EPOCHS,
        imgsz=416,  # Image size for training (adjustable)
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,  # Initial learning rate
        lrf=0.001,  # Learning rate factor for scheduler
        save_period=1,  # Save weights after every epoch
        project=MODEL_SAVE_DIR,  # Directory for model output
        name="yolov11_pallets",  # Subdirectory name for this experiment
        device="cuda"  # Use GPU for training
    )
    print("Training completed!")

    # Save the trained YOLOv11 model
    trained_model_path = os.path.join(MODEL_SAVE_DIR, "yolov11_pallets.pt")
    model.save(trained_model_path)
    print(f"Trained model saved at: {trained_model_path}")

# Start training
train_yolo()
