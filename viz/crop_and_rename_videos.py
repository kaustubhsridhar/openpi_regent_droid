import os 
from viz_info import video_paths
import cv2
from PIL import Image
import numpy as np
new_fol = 'videos_cropped_simplified_dont_delete'
os.makedirs(new_fol, exist_ok=True)

for key_of_method_task, location in video_paths.items():
    key_of_task_method = f'{key_of_method_task.split("_")[-1]}_{"_".join(key_of_method_task.split("_")[:-1])}'
    print(f'{key_of_method_task} -> cropped {key_of_task_method}')
    if os.path.exists(f'{new_fol}/{key_of_task_method}.mp4'):
        print(f'already exists\n')
        continue

    # read video from location
    cap = cv2.VideoCapture(location)

    # iteratie over frames
    frame_count = 0
    cropped_video = []
    while True:
        # Read a frame
        ret, frame = cap.read()
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        # Crop frame to middle one third in width if regent, first half if pi0
        frame_image = Image.fromarray(frame)
        if 'regent' in key_of_method_task or 'rnp' in key_of_method_task:
            cropped_frame = frame_image.crop((frame_image.width//3, 0, frame_image.width//3*2, frame_image.height))
        else:
            cropped_frame = frame_image.crop((0, 0, frame_image.width//2, frame_image.height))
        cropped_video.append(cropped_frame)

    # Release the video capture object
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # save the cropped video to new_fol/{key_of_task_method}.mp4
    cropped_video = [np.array(frame) for frame in cropped_video]
    # write the cropped video to a new file at the same fps as when I read it
    print(f'fps before tripling: {fps}')
    fps *= 3 # multiply the fps
    out = cv2.VideoWriter(f'{new_fol}/{key_of_task_method}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (cropped_video[0].shape[1], cropped_video[0].shape[0]))
    for frame in cropped_video:
        out.write(frame)
    out.release()
    
    print(f'cropped\n')
