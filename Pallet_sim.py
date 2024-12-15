import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
import numpy as np

# Configuration
MODEL_PATH = "path/to/model"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4
CLASS_NAMES = ["pallet"]
YOLO_IMAGE_SIZE = (640, 480)  # Resize dimensions for YOLO model processing

class DepthCameraNode(Node):
    def __init__(self):
        super().__init__("depth_camera_node")
        self.get_logger().info("Depth Camera Node initialized.")

        # Subscribe to the depth camera's 2D image topic
        self.create_subscription(Image, "/depth_camera/image_raw", self.image_callback, 10)

        self.create_subscription(CameraInfo, "/depth_camera/camera_info", self.camera_info_callback, 10)

        # YOLO model
        self.model = YOLO(MODEL_PATH)

        # CV Bridge for converting ROS images to OpenCV
        self.bridge = CvBridge()

        # Camera parameters (optional)
        self.camera_matrix = None
        self.dist_coeffs = None

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape((3, 3))  # Intrinsic camera matrix
        self.dist_coeffs = np.array(msg.d)  # Distortion coefficients
        self.get_logger().info(f"Camera matrix: {self.camera_matrix}")
        self.get_logger().info(f"Distortion coefficients: {self.dist_coeffs}")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.get_logger().info(f"Received image with encoding: {msg.encoding} and shape: {frame.shape}")

            # Check if the image is single-channel (grayscale or depth)
            if len(frame.shape) == 2:
                # Visualize depth data (convert to 8-bit for display)
                frame_display = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                frame_display = cv2.applyColorMap(frame_display, cv2.COLORMAP_JET)  # Add color for visualization
                frame_bgr = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = frame  # Already in BGR format

            # Display the image to ensure it's being received correctly
            cv2.imshow("Depth Camera Feed", frame_bgr)

            # Resize the frame for YOLO model
            resized_frame = cv2.resize(frame_bgr, YOLO_IMAGE_SIZE)

            # Perform YOLO detection
            results = self.model.predict(resized_frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

            # Draw detections on the original frame
            self.display_detections(frame_bgr, results)

            cv2.waitKey(1)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

    def display_detections(self, frame, results):
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
            class_id = int(result.cls[0])  # Class ID
            confidence = result.conf[0]  # Confidence score

            if confidence >= CONFIDENCE_THRESHOLD:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display label and confidence
                label = f"{CLASS_NAMES[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame with detections
        cv2.imshow("YOLO Detection", frame)


def main(args=None):
    rclpy.init(args=args)
    node = DepthCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Depth Camera Node.")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

