import os   
import cv2
from PIL import Image
from viz_info import video_paths

# read videos and convert to frames
methods = ['pi0', 'regent', 'regent_finetune']
tasks = ['idli_plate', 'pokeball', 'squeegee', 'bagel', 'lever', 'door']

# Create output directory for frames
frames_dir = 'frames_dont_delete'
os.makedirs(frames_dir, exist_ok=True)

for task in tasks:
        # Create task directory
    task_dir = os.path.join(frames_dir, task)
    os.makedirs(task_dir, exist_ok=True)
    
    for method in methods:
        video_path = video_paths[f'{method}_{task}']
        
        # Skip if video path is empty
        if not video_path:
            print(f"Skipping {method}_{task}: No video path provided")
            continue
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found at {video_path}")
            continue
        
        # Create method directory within task directory
        method_dir = os.path.join(task_dir, method)
        # if method_dir already exists, skip
        if os.path.exists(method_dir):
            print(f"We did already and hence skipping {method}_{task}: Method directory already exists")
            continue
        # otherwise create it
        os.makedirs(method_dir, exist_ok=True)
        
        print(f"Processing video: {method}_{task}")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue
        
        frame_count = 0
        
        while True:
            # Read a frame
            ret, frame = cap.read()
            
            # Break the loop if we've reached the end of the video
            if not ret:
                break
            
            # Save frame as JPEG file
            frame_path = os.path.join(method_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            frame_count += 1
        
        # Release the video capture object
        cap.release()
        
        print(f"Extracted {frame_count} frames from {method}_{task}")

print("Frame extraction complete")