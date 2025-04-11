import os   
import cv2
from PIL import Image

video_paths = {
'pi0_idli_plate' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_idli_plate/right_move_the_idli_plate_to_the_right_2025_03_14_17:17:20.mp4.mp4', 
'regent_idli_plate' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_pokeball/ckpt5400longer-0317poketraydata-4_pick_up_the_poke_ball_and_put_it_in_the_tray_2025_03_17_13:56:25.mp4',
'regent_finetune_idli_plate' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_finetune_idli_plate/finetune-idli-plate-ckpt999-DYNAMIC_move_the_idli_plate_to_the_right_2025_03_24_09:30:37.mp4',
#
'pi0_pokeball' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_pokeball/right_pick_up_the_pokeball_and_put_it_in_the_tray_2025_04_08_12:43:48.mp4.mp4',
'regent_pokeball' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_pokeball/ckpt5400longer-0317poketraydata-4_pick_up_the_poke_ball_and_put_it_in_the_tray_2025_03_17_13:56:25.mp4',
'regent_finetune_pokeball' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_finetune_pokeball/finetune-pokeball-ckpt999-DYNAMIC_pick_up_the_poke_ball_and_put_it_in_the_tray_2025_03_24_10:55:26.mp4',
#
'pi0_squeegee' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_squeegee/right_move_the_squeegee_to_the_right_and_try_to_drag_it_2025_03_27_03:24:15.mp4.mp4', 
'regent_squeegee' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_squeegee/ckpt5400longer-0327squeegeedata-6_move_the_squeegee_to_the_right_and_try_to_drag_it_2025_03_27_02:21:50.mp4',
'regent_finetune_squeegee' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_finetune_squeegee/finetune-squeegee-ckpt999-DYNAMIC3_move_the_squeegee_to_the_right_and_try_to_drag_it_2025_03_27_06:47:53.mp4',
#
'pi0_bagel' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_toaster_bagel/right_pick_up_the_bagel_and_put_it_in_the_toaster_2025_04_07_16:35:40.mp4.mp4',
'regent_bagel' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_toaster_bagel/bagel_8_pick_up_the_bagel_and_put_it_in_the_toaster_2025_04_08_13:39:58.mp4',
'regent_finetune_bagel' : f'',
#
'pi0_lever' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_toaster_lever/right_push_the_lever_on_the_toaster_2025_04_07_16:43:19.mp4.mp4', # aimlessly wanders
'regent_lever' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_toaster_lever/lever_2_push_the_lever_on_the_toaster_2025_04_07_19:16:13.mp4',
'regent_finetune_lever' : f'',
#
'pi0_door' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_door_shelf/right_open_the_door_of_the_bottom_locker_2025_04_07_18:12:28.mp4.mp4', # aimlessly wanders
'regent_door' : f'',
'regent_finetune_door' : f'',
}

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