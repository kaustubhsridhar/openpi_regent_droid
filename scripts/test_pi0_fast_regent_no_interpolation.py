import openpi.training.config as _config
import openpi.policies.policy_config as _policy_config
import openpi.models.model as _model
import numpy as np
from openpi.policies.utils import init_logging

init_logging()

# params
config_name = "pi0_fast_droid_regent"
exp_name = "tenth_try_firstnonlora"
checkpoint_step = 300
demos_dir = "regent_droid_preprocessing/collected_demos/2025-03-09_bowlx0y0"

# setup
config = _config.get_config(config_name)
checkpoint_dir = f"checkpoints/{config_name}/{exp_name}/{checkpoint_step}"

# create a trained policy
policy = _policy_config.create_trained_regent_policy(train_config=config, checkpoint_dir=checkpoint_dir, demos_dir=demos_dir)

# load a test demo
demo_path = "regent_droid_preprocessing/collected_demos/2025-03-09_bowlx1y0/2025-03-09_08-20-01_bowlx1y0_pokeballx0y0_recovery/processed_demo.npz"
demo = np.load(demo_path)

# run inference on a example 
camera = "left"
example = {
        "query_top_image": demo[f"top_image"][0],
        "query_right_image": demo[f"right_image"][0],
        "query_wrist_image": demo[f"wrist_image"][0],
        "query_state": demo["state"][0],
        "query_prompt": demo["prompt"].item(),
        "camera": camera
    }
action_chunk = policy.infer(example)["query_actions"] # camera can be "left" or "right"
