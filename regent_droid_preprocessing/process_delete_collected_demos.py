import numpy as np
import json 
import h5py
import os
import argparse
from PIL import Image
from openpi.policies.utils import embed_with_batches, load_dinov2
from openpi_client.image_tools import resize_with_pad
import logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
args = parser.parse_args()

# get current directory and append the dir argument to get demo_dir
current_dir = os.path.dirname(os.path.abspath(__file__)) # get current directory
demo_dir = f"{current_dir}/{args.dir}"
logger.info(f'absolute path of the {demo_dir=}')

# get all the folders (demos) in the demo_dir
demo_folders = [f"{demo_dir}/{f}" for f in os.listdir(demo_dir) if os.path.isdir(f"{demo_dir}/{f}")]
logger.info(f'number of demo folders: {len(demo_folders)}')

# iterate over the demo_folders and read the trajectory.h5 files and the frames
for demo_folder in demo_folders:
    filename = f'processed_demo'
    if os.path.exists(f"{demo_folder}/{filename}.npz"):
        os.remove(f"{demo_folder}/{filename}.npz")




    