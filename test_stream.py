import cv2
import torch
import sys
import os
import numpy as np

# 1. Setup Path
REPO_SRC_PATH = os.path.join(os.getcwd(), 'src/Depth-Anything-3/src')
sys.path.append(REPO_SRC_PATH)

try:
    from depth_anything_3.api import DepthAnything3
except ImportError as e:
    print(f"Error importing DepthAnything3: {e}")
    sys.exit(1)

# 2. Setup Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on: {DEVICE}")

# 3. Load Model (Using the official API)
try:
    print("Loading Depth Anything V3 (Small)...")
    # Note: We use the exact string from your docs
    model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
    model = model.to(device=DEVICE)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

# 4. Webcam Loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Convert BGR (OpenCV) to RGB (DepthAnything expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 5. Inference using the high-level API
    # The API expects a list of images (even if just one)
    # We pass None to export_dir to avoid saving files to disk (speed up)
    prediction = model.inference(
    [rgb_frame], 
    export_dir=None,    # Keep this None to avoid saving files
    export_format="npz" # Just a string to satisfy the API check
)
    # 6. Extract Depth
    # The documentation says prediction.depth is [N, H, W]
    # We passed 1 image, so we take index 0
    depth_map = prediction.depth[0] 

    # Normalize for visualization (0-255)
    # depth_map is float32, usually relative depth
    if depth_map.max() > depth_map.min():
        depth_display = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    else:
        depth_display = depth_map * 0 # Handle flat scenes
        
    depth_display = depth_display.astype('uint8')
    depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_MAGMA)

    # Show results
    cv2.imshow('Depth Anything V3', depth_color)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
