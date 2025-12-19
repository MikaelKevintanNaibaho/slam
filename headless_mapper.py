import cv2
import torch
import sys
import os
import numpy as np
import socket
import struct
import time
import open3d as o3d

# --- CONFIGURATION ---
VIDEO_PATH = "tum1.mp4"
HOST = '127.0.0.1'
PORT = 8080

# Tighter depth range for cleaner desk maps
DEPTH_SCALE = 3000.0        # 3 meters max range
DEPTH_TRUNC = 2.5           # Hard cut-off at 2.5m

LOOP_VIDEO = False           

# 1. Setup AI Model
REPO_SRC_PATH = os.path.join(os.getcwd(), 'src/Depth-Anything-3/src')
sys.path.append(REPO_SRC_PATH)
try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("Error importing DepthAnything3")
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading Model on {DEVICE}...")
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to(DEVICE).eval()

# 2. Setup The Invisible Map
print("Initializing Dense Volume...")
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.015,     # 1.5cm voxels (High Detail)
    sdf_trunc=0.05,         # Thickness of surface
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 517.3, 516.5, 318.6, 255.3)

# 3. Connect to SLAM
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.connect((HOST, PORT))
    print(f"Connected to SLAM Server at {HOST}:{PORT}")
except:
    print("Failed to connect. Is ./socket_slam running?")
    sys.exit(1)

# 4. Setup Video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    sys.exit(1)

# Get video FPS to calculate correct timestamps
fps = cap.get(cv2.CAP_PROP_FPS)
if fps < 1: fps = 30.0

start_time = time.time()
frame_id = 0

print("=== RUNNING V2 (Edge Filtered)! Press 'q' to stop ===")

try:
    while True:
        ret, raw_frame = cap.read()
        
        if not ret:
            if LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        # Resize to match SLAM Calibration
        frame = cv2.resize(raw_frame, (640, 480))
        h, w = frame.shape[:2]

        # --- AI DEPTH ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            prediction = model.inference([rgb_frame], export_dir=None, export_format="npz")
        
        depth = prediction.depth[0]
        
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # --- CLEANUP STEP 1: Percentile Clipping (Remove Spikes) ---
        d_min = np.percentile(depth, 2)
        d_max = np.percentile(depth, 98)
        depth = np.clip(depth, d_min, d_max)
        depth_normalized = (depth - d_min) / (d_max - d_min + 1e-6)

        # --- CLEANUP STEP 2: Edge Eraser (Flying Pixel Removal) ---
        # Calculate gradients (where depth changes fast)
        grad_x = cv2.Sobel(depth_normalized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_normalized, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        
        # Mask out edges that are too sharp (threshold 0.5 usually works well)
        edge_mask = magnitude > 0.5
        depth_normalized[edge_mask] = 0
        
        # --- CLEANUP STEP 3: Bilateral Filter (Smooth Surfaces) ---
        # Convert to float32 for filtering
        d_float = depth_normalized.astype(np.float32)
        d_filtered = cv2.bilateralFilter(d_float, 5, 0.1, 0.1)
        
        depth_mm = (d_filtered * DEPTH_SCALE).astype(np.uint16)

        # --- SEND TO C++ ---
        # FIX: Use calculated timestamp, NOT system clock
        sim_timestamp = frame_id / fps
        header = struct.pack('=iid', w, h, sim_timestamp)
        
        try:
            sock.sendall(header)
            sock.sendall(frame.tobytes())
            sock.sendall(depth_mm.tobytes())

            status_data = sock.recv(4)
            if not status_data: break
            status = struct.unpack('i', status_data)[0]

            pose_data = sock.recv(64)
            if not pose_data: break
        except BrokenPipeError:
            print("Server disconnected.")
            break
        
        if status == 1:
            pose_flat = struct.unpack('16f', pose_data)
            T_cw = np.array(pose_flat).reshape(4, 4)
            
            try:
                T_wc = np.linalg.inv(T_cw)
            except:
                continue

            o3d_color = o3d.geometry.Image(rgb_frame)
            o3d_depth = o3d.geometry.Image(depth_mm)
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth, 
                depth_scale=1000.0, 
                depth_trunc=DEPTH_TRUNC, 
                convert_rgb_to_intensity=False
            )

            volume.integrate(rgbd, intrinsic, T_wc)

        # Preview
        cv2.imshow('Clean Mapper V2', frame)
        cv2.imshow('Depth Filtered', (d_filtered * 255).astype(np.uint8)) # See what the map sees
        
        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")

print("Extracting final mesh...")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh("final_map_v2.ply", mesh)
print("Saved to final_map_v2.ply")

sock.close()
cap.release()
cv2.destroyAllWindows()

# Auto-open viewer
o3d.visualization.draw_geometries([mesh], window_name="Result", width=960, height=540)
