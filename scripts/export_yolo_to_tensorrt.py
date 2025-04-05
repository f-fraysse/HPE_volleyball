import os
from ultralytics import YOLO
from paths import MODEL_DIR

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

print("Exporting YOLO11n to TensorRT format...")
# Export detection model
det_model = YOLO('yolo11m.pt')
success = det_model.export(format='engine', device=0)  # Exports to 'yolo11n.engine'
print(f"Detection model export {'successful' if success else 'failed'}")

print("Exporting YOLO11n-pose to TensorRT format...")
# Export pose model
pose_model = YOLO('yolo11m-pose.pt')
success = pose_model.export(format='engine', device=0)  # Exports to 'yolo11n-pose.engine'
print(f"Pose model export {'successful' if success else 'failed'}")

print("Export complete. TensorRT engine files should be available in the current directory.")
