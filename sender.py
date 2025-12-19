import cv2
import torch
import sys
import os
import numpy as np
import socket
import struct
import time

# 1. Setup Model
REPO_SRC_PATH = os.path.join(os.getcwd(), 'src/Depth-Anything-3/src')
sys.path.append(REPO_SRC_PATH)

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("Error importing DepthAnything3")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to(DEVICE).eval()

# 2. Connect to C++ Server
HOST = '127.0.0.1'
PORT = 8080

print(f"Connecting to C++ SLAM server at {HOST}:{PORT}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((HOST, PORT))
    print("Connected!")
except ConnectionRefusedError:
    print("Error: Could not connect. Make sure './socket_slam' is running FIRST.")
    sys.exit(1)

# 3. Webcam Loop
cap = cv2.VideoCapture(0)
start_time = time.time()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- FIX 1: Get Dimensions EARLY ---
    h, w = frame.shape[:2]

    # A. Inference
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        prediction = model.inference([rgb_frame], export_dir=None, export_format="npz")
    
    # --- FIX 2: Extract Depth BEFORE checking it ---
    depth = prediction.depth[0]

    # --- FIX 3: Resize Check ---
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Normalize 0..1 -> 0..5000mm (16-bit integer)
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth_mm = (depth_normalized * 5000).astype(np.uint16)
   
    # C. Send to C++
    # Header: width(4), height(4), timestamp(8)
    timestamp = time.time() - start_time
    
    # Standard packing (=) to ensure C++ reads it correctly
    header = struct.pack('=iid', w, h, timestamp)
    
    try:
        sock.sendall(header)
        sock.sendall(frame.tobytes())        # RGB (Raw bytes)
        sock.sendall(depth_mm.tobytes())     # Depth (Raw bytes)
    except BrokenPipeError:
        print("Server disconnected.")
        break


    # D. Save Data for Dense Mapping
    # Save filenames as timestamps to match the trajectory file
    timestamp_str = f"{timestamp:.6f}"

    # Save RGB
    cv2.imwrite(f"data/rgb/{timestamp_str}.png", frame)

    # Save Depth (Raw 16-bit)
    cv2.imwrite(f"data/depth/{timestamp_str}.png", depth_mm)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Sent & Saved frame {frame_count}...")
        # Visualization (Optional)
        cv2.imshow('Sending to SLAM...', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

sock.close()
cap.release()
cv2.destroyAllWindows()
