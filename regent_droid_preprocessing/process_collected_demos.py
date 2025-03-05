import numpy as np
import json 
import h5py
import os
import argparse
from PIL import Image
from utils import embed_with_batches, load_policy
from openpi_client.image_tools import resize_with_pad
import logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True)
parser.add_argument("--prompts", nargs="+", type=str, required=True)
args = parser.parse_args()

# load a policy (only for embedding images)
policy_name = "pi0_fast_droid"
policy = load_policy(policy_name)
logger.info(f'loaded {policy_name} policy (only for embedding images)')

# get current directory and append the dir argument to get demo_dir
current_dir = os.path.dirname(os.path.abspath(__file__)) # get current directory
demo_dir = f"{current_dir}/{args.dir}"
logger.info(f'absolute path of the {demo_dir=}')

# get all the folders (demos) in the demo_dir
demo_folders = [f"{demo_dir}/{f}" for f in os.listdir(demo_dir) if os.path.isdir(f"{demo_dir}/{f}")]
logger.info(f'number of demo folders: {len(demo_folders)}')

# iterate over the demo_folders and read the trajectory.h5 files and the frames
for demo_folder in demo_folders:
    processed_demo = {}
    logger.info(f'processing {demo_folder=}')
    traj_h5 = h5py.File(f"{demo_folder}/trajectory.h5", 'r')

    skip_bools = traj_h5["observation"]["timestamp"]["skip_action"][:]
    keep_bools = ~skip_bools

    obs_gripper_pos = traj_h5["observation"]["robot_state"]["gripper_position"][:].reshape(-1, 1)[keep_bools]
    act_gripper_pos = traj_h5["action"]["gripper_position"][:].reshape(-1, 1)[keep_bools]
    obs_joint_pos = traj_h5["observation"]["robot_state"]["joint_positions"][keep_bools]
    act_joint_vel = traj_h5["action"]["joint_velocity"][keep_bools]
    
    processed_demo["state"] = np.concatenate([obs_joint_pos, obs_gripper_pos], axis=1)
    processed_demo["actions"] = np.concatenate([act_joint_vel, act_gripper_pos], axis=1)
    num_steps = processed_demo["state"].shape[0]
    assert processed_demo["state"].shape == processed_demo["actions"].shape == (num_steps, 8)

    for camera_name, key in zip(['hand_camera', 'varied_camera_1', 'varied_camera_2'], ['wrist_image', 'left_image', 'right_image']):
        frames_dir = f"{demo_folder}/recordings/frames/{camera_name}"
        logger.info(f'{frames_dir=}')
        frames = [f"{frames_dir}/{f}" for f in os.listdir(frames_dir)]
        assert len(frames) == num_steps, f'{len(frames)=} {num_steps=}'
        frames = [np.array(Image.open(frame)) for frame in frames]
        frames = np.stack(frames, axis=0)
        assert frames.shape == (num_steps, 720, 1280, 3) and frames.dtype == np.uint8, f'{frames.shape=} {frames.dtype=}'
        frames = resize_with_pad(frames, 224, 224)
        assert frames.shape == (num_steps, 224, 224, 3) and frames.dtype == np.uint8, f'{frames.shape=} {frames.dtype=}'
        processed_demo[key] = frames
        
        if key == 'wrist_image':
            embeddings = embed_with_batches(frames, policy, batch_size=400)
            assert embeddings.shape == (num_steps, 2048), f'{embeddings.shape=}'
            processed_demo[f"embeddings"] = embeddings

    # randomly sample a prompt from the prompts
    prompt = np.random.choice(args.prompts)
    processed_demo["prompt"] = prompt

    # save the processed episode as a npz file
    np.savez(f"{demo_folder}/processed_demo.npz", **processed_demo)




    