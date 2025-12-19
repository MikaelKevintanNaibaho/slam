import cv2
import torch
import numpy as np
import socket
import struct
import open3d as o3d
import os
import sys

# --- FIX IMPORT PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_SRC_PATH = os.path.join(BASE_DIR, 'src', 'Depth-Anything-3', 'src')
sys.path.append(REPO_SRC_PATH)
from depth_anything_3.api import DepthAnything3

# --- CONFIG ---
 
VIDEO_PATH = "tum1.mp4"
INTRINSIC_OBJ = o3d.camera.PinholeCameraIntrinsic(640, 480, 517.3, 516.5, 318.6, 255.3)
intrinsic_matrix = INTRINSIC_OBJ.intrinsic_matrix

# --- TSDF VOLUME (The "Solid Surface" Engine) ---
# sdf_trunc: defines how much 'smoothing' happens near the surface. 
# 0.04 (4cm) is good for detail.
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.04,
    sdf_trunc=0.08,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

# 1. Model & Socket
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to('cuda').eval()
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 8080))

cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0

print("=== STARTING TSDF SOLID SURFACE RECONSTRUCTION ===")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]

        # A. AI Prediction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            pred = model.inference([rgb], export_dir=None, export_format="npz")
        skin_relative = 1.0 / (cv2.resize(pred.depth[0], (w, h)) + 1e-6)

        # B. SLAM Sync
        header = struct.pack('=iid', w, h, frame_id / 20.0)
        sock.sendall(header + frame.tobytes() + (skin_relative.astype(np.float32)).tobytes()[:(w*h*2)]) # Dummy depth

        # C. Receive Pose & Skeleton
        status = struct.unpack('i', sock.recv(4))[0]
        pose = np.frombuffer(sock.recv(64), dtype=np.float32).reshape(4,4, order='F')
        n_pts = struct.unpack('i', sock.recv(4))[0]
        
        if n_pts > 0:
            skeleton = np.frombuffer(sock.recv(n_pts * 12), dtype=np.float32).reshape(-1, 3)
            
            if status == 1:
                # D. Scale Fusion
                ai_v, slam_v = [], []
                for u, v, z_slam in skeleton:
                    ui, vi = int(u), int(v)
                    if 0 <= ui < w and 0 <= vi < h:
                        ai_v.append(skin_relative[vi, ui])
                        slam_v.append(z_slam)
                
                if len(ai_v) > 10:
                    scale = np.median(np.array(slam_v) / np.array(ai_v))
                    real_depth = skin_relative * scale
                    
                    # E. INTEGRATE INTO VOLUME (Solid Surface logic)
                    color_o3d = o3d.geometry.Image(rgb)
                    depth_o3d = o3d.geometry.Image((real_depth * 1000).astype(np.uint16))
                    
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color_o3d, depth_o3d, depth_scale=1000.0, 
                        depth_trunc=12.0, convert_rgb_to_intensity=False
                    )
                    
                    # This "melts" the frame into the existing 3D structure
                    volume.integrate(rgbd, INTRINSIC_OBJ, np.linalg.inv(pose))

        print(f"Frame {frame_id} | Scale: {scale if 'scale' in locals() else 0:.2f}", end='\r')
        frame_id += 1
        cv2.imshow('Warehouse Feed', frame)
        if cv2.waitKey(1) == ord('q'): break

except KeyboardInterrupt:
    pass

# F. EXTRACT FINAL MESH
print("\nExtracting Mesh (Marching Cubes)...")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()

o3d.io.write_triangle_mesh("solid_warehouse_mesh.ply", mesh)
print("Saved solid_warehouse_mesh.ply")
o3d.visualization.draw_geometries([mesh])
