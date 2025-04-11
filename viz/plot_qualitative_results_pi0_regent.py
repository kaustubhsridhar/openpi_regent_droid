import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
from viz_info import chosen_frames, tasks, methods


final_dir = 'frames_dont_delete_concat'
os.makedirs(final_dir, exist_ok=True)

for task in tasks:
    task_dir = os.path.join('frames_dont_delete', task)
    for method in methods:
        method_dir = os.path.join(task_dir, method)
        final_path = os.path.join(final_dir, f'{task}_{method}.jpg')
        if os.path.exists(final_path):
            print(f'We already have this one and skipping {method}_{task}')
            continue
        # read the chosen frames of this task and method
        chosen_frames_curr = chosen_frames[f'{method}_{task}']
        if len(chosen_frames_curr) in [0, 1]:
            print(f'No frames chosen yet and skipping {method}_{task}')
            continue
        chosen_frames_curr = [os.path.join(method_dir, f'frame_{frame_count:04d}.jpg') for frame_count in chosen_frames_curr]
        # read the images
        images = [Image.open(frame) for frame in chosen_frames_curr]
        # split the image to take first half if method is pi0 and middle third otherwise
        if method == 'pi0':
            images = [image.crop((0, 0, image.width // 2, image.height)) for image in images]
        else:
            images = [image.crop((image.width // 3, 0, image.width // 3 * 2, image.height)) for image in images]
        # concat the images side by side with a gap of 10 pixels between them and save to final_path
        final_image = Image.new('RGB', (images[0].width * len(images) + (len(images) - 1) * 10, images[0].height))
        for i, image in enumerate(images):
            final_image.paste(image, (i * (image.width + 10), 0))
        final_image.save(final_path)
        print(f'Done with {method}_{task}')