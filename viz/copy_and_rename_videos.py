import os 
from viz_info import video_paths

new_fol = 'videos_copy_simplified_dont_delete'
os.makedirs(new_fol, exist_ok=True)

for key_of_method_task, location in video_paths.items():
    key_of_task_method = f'{key_of_method_task.split("_")[-1]}_{"_".join(key_of_method_task.split("_")[:-1])}'
    print(f'{key_of_method_task} -> {key_of_task_method}')
    if os.path.exists(f'{new_fol}/{key_of_task_method}.mp4'):
        print(f'already exists\n')
        continue
    os.system(f'cp {location} {new_fol}/{key_of_task_method}.mp4')
    print(f'copied\n')
