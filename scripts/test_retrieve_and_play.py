import openpi.training.config as _config
import openpi.policies.policy_config as _policy_config
import openpi.models.model as _model
import numpy as np
from openpi.policies.utils import init_logging

init_logging()

# params
config_name = "retrieve_and_play"
demos_dir = "regent_droid_preprocessing/collected_demos/2025-03-08"

# create a trained policy
policy = _policy_config.create_retrieve_and_play_policy(demos_dir=demos_dir)

# load a test demo
demo_path = "regent_droid_preprocessing/collected_demos/2025-03-08/2025-03-08_20-06-09/processed_demo.npz"
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
action_chunk = policy.infer(example)["query_actions"] # camera can be "left" or "right"
