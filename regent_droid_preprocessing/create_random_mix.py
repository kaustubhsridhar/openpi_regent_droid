import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source_folder_name", type=str, default="collected_demos_training")
parser.add_argument("--target_folder_name", type=str, default="collected_demos_training/2025-04-18_random_mix")
args = parser.parse_args()

os.makedirs(args.target_folder_name, exist_ok=True)

# get all folders in the source folder
source_folder_path = os.path.join(os.getcwd(), args.source_folder_name)
folders = [f for f in os.listdir(source_folder_path) if os.path.isdir(os.path.join(source_folder_path, f))]

# in each folder, randomly select one subfolder
for folder in folders:
    subfolders = [f for f in os.listdir(os.path.join(source_folder_path, folder)) if os.path.isdir(os.path.join(source_folder_path, folder, f))]
    selected_subfolder = random.choice(subfolders)
    os.system(f"cp -r {os.path.join(source_folder_path, folder, selected_subfolder)} {os.path.join(args.target_folder_name, selected_subfolder.replace('_', '-')+'-from-'+folder)}")
