import cv2
import os
import glob

# Path to the RGB images in the dataset
image_folder = 'tum_data/rgb'
video_name = 'tum1.mp4'

print("Finding images...")
# TUM images are named by timestamp, so we sort them to keep order
images = sorted(glob.glob(os.path.join(image_folder, "*.png")))

if not images:
    print("Error: No images found! Did you extract the folder correctly?")
    exit()

# Read first image to get size
frame = cv2.imread(images[0])
height, width, layers = frame.shape

print(f"Creating video {video_name} from {len(images)} frames...")
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

for image in images:
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()
print("Done! You now have 'tum1.mp4'")
