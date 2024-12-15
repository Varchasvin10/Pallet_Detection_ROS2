import cv2
from ultralytics import YOLO

# Configuration
MODEL_PATH = "Path/to/model"  # Replace with your YOLO model path
CLASS_NAMES = ["pallet"]  # Replace with the class names your model was trained on
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

def detect_pallets(image_path):
    # Load YOLO model
    model = YOLO(MODEL_PATH)

    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot open image at {image_path}")
        return

    # Run inference
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

    # Draw detections on the image
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        class_id = int(result.cls[0])  # Class ID
        confidence = result.conf[0]  # Confidence score

        if confidence >= CONFIDENCE_THRESHOLD:
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display label and confidence
            label = f"{CLASS_NAMES[class_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output image
    cv2.imshow("Pallet Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your image path
    IMAGE_PATH = "image/path"
    detect_pallets(IMAGE_PATH)

