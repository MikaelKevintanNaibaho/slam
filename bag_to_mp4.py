import sys
import cv2
import numpy as np
from pathlib import Path  # <--- REQUIRED FIX
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# --- USAGE ---
# python bag_to_mp4.py MH_01_easy.bag

if len(sys.argv) < 2:
    print("Usage: python bag_to_mp4.py <path_to_bag_file>")
    sys.exit(1)

# FIX: Convert string to Path object
bag_path = Path(sys.argv[1])
output_file = "warehouse.mp4"
target_topic = "/cam0/image_raw" 

print(f"Opening {bag_path}...")

# 1. Setup Type Store
typestore = get_typestore(Stores.ROS1_NOETIC)

out = None
frame_count = 0

try:
    # 2. Use AnyReader
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        
        # Check for topic
        connections = [x for x in reader.connections if x.topic == target_topic]
        if not connections:
            print(f"Error: Topic '{target_topic}' not found.")
            print("Available topics:", [c.topic for c in reader.connections])
            sys.exit(1)

        print(f"Extracting video from {target_topic}...")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            # 3. Deserialize
            msg = reader.deserialize(rawdata, connection.msgtype)

            # 4. Process Image
            width = msg.width
            height = msg.height
            data = np.frombuffer(msg.data, dtype=np.uint8)
            img = data.reshape((height, width))
            
            # Convert to BGR for Video
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if out is None:
                print(f"Video Resolution: {width}x{height}")
                out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
            
            out.write(img_bgr)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...", end='\r')

    print(f"\nSUCCESS! Saved {frame_count} frames to '{output_file}'")

except Exception as e:
    print(f"\nFAILED: {e}")
finally:
    if out: out.release()
