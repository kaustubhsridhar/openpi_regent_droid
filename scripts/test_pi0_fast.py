from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np
from openpi.policies.utils import init_logging

init_logging()

config = config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# load a test demo
demo_path = "regent_droid_preprocessing/collected_demos/2025-03-04_00-48-37/processed_demo.npz"
demo = np.load(demo_path)

# Run inference on a example 
camera = "right"
example = {
    "observation/exterior_image_1_left": demo[f"{camera}_image"][0],
    "observation/wrist_image_left": demo[f"wrist_image"][0],
    "observation/joint_position": demo["state"][0][:7],
    "observation/gripper_position": demo["state"][0][7:8],
    "prompt": demo["prompt"].item(),
}

action_chunk = policy.infer(example)["actions"]
print(f'{action_chunk.shape=}')
print(f'{action_chunk=}')
