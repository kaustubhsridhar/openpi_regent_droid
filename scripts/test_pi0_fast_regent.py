import openpi.training.config as _config
import openpi.policies.policy_config as _policy_config
import openpi.models.model as _model
import numpy as np
from openpi.policies.utils import init_logging

init_logging()

# params
config_name = "pi0_fast_droid_regent_with_interpolation"
exp_name = "first_try_with_interpolation"
checkpoint_step = 3000
demos_dir = "regent_droid_preprocessing/collected_demos/2025-03-04"

# setup
config = _config.get_config(config_name)
checkpoint_dir = f"checkpoints/{config_name}/{exp_name}/{checkpoint_step}"

# create a trained policy
policy = _policy_config.create_trained_regent_policy(train_config=config, checkpoint_dir=checkpoint_dir, demos_dir=demos_dir)

# load a test demo
demo_path = "regent_droid_preprocessing/collected_demos/2025-03-04_00-48-37/processed_demo.npz"
demo = np.load(demo_path)

# run inference on a example 
camera = "right"
example = {
        "query_image": demo[f"{camera}_image"][0],
        "query_wrist_image": demo[f"wrist_image"][0],
        "query_state": demo["state"][0],
        "query_prompt": demo["prompt"].item(),
        "camera": camera
    }
action_chunk = policy.infer(example)["actions"] # camera can be "left" or "right"
print(f'{action_chunk.shape=}')
print(f'{action_chunk=}')
