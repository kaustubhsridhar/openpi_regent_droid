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

# run inference on a dummy example.
example = {
        "query_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "query_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "query_state": np.random.rand(8), # joint position (7) + gripper position (1)
        "query_prompt": "pick up the pokeball and put it in the bowl",
        "camera": "right"
    }
action_chunk = policy.infer(example)["actions"] # camera can be "left" or "right"
print(f'{action_chunk.shape=}')
print(f'{action_chunk=}')
