import json 
import os
import numpy as np 

norm_stats_basic_file = "assets/droid_basic/norm_stats_simple.json"
new_norm_stats_file = "assets/pi0_fast_droid_regent/droid/norm_stats.json"
os.makedirs("assets/pi0_fast_droid_regent/droid", exist_ok=True)
new_norm_stats_file_with_interpolation = "assets/pi0_fast_droid_regent_with_interpolation/droid/norm_stats.json"
os.makedirs("assets/pi0_fast_droid_regent_with_interpolation/droid", exist_ok=True)
new_norm_stats_file_with_interpolation_longer_act_horizon = "assets/pi0_fast_droid_regent_with_interpolation_longer_act_horizon/droid/norm_stats.json"
os.makedirs("assets/pi0_fast_droid_regent_with_interpolation_longer_act_horizon/droid", exist_ok=True)

norm_stats_basic = json.load(open(norm_stats_basic_file, 'r'))
num_retrieved = 5
action_horizon = 10
longer_action_horizon = 15

new_norm_stats = {"norm_stats": {}}
new_norm_stats_longer = {"norm_stats": {}}
for key in norm_stats_basic["norm_stats"]:
    if key == "actions":
        sub_keys = norm_stats_basic["norm_stats"]["actions"].keys()
        action_chunk = {subk: [] for subk in sub_keys}
        action_chunk_longer = {subk: [] for subk in sub_keys}
        for sub_key in sub_keys:
            for i in range(action_horizon):
                action_chunk[sub_key].extend(norm_stats_basic["norm_stats"]["actions"][sub_key])
            for i in range(longer_action_horizon):
                action_chunk_longer[sub_key].extend(norm_stats_basic["norm_stats"]["actions"][sub_key])
        

    for i in range(num_retrieved):
        prefix = f"retrieved_{i}_"
        new_norm_stats["norm_stats"][f"{prefix}{key}"] = action_chunk if key == "actions" else norm_stats_basic["norm_stats"][key]
        new_norm_stats_longer["norm_stats"][f"{prefix}{key}"] = action_chunk_longer if key == "actions" else norm_stats_basic["norm_stats"][key]
    prefix = f"query_"
    new_norm_stats["norm_stats"][f"{prefix}{key}"] = action_chunk if key == "actions" else norm_stats_basic["norm_stats"][key]
    new_norm_stats_longer["norm_stats"][f"{prefix}{key}"] = action_chunk_longer if key == "actions" else norm_stats_basic["norm_stats"][key]


if not os.path.exists(new_norm_stats_file):
    print(f'writing {new_norm_stats_file}')
    json.dump(new_norm_stats, open(new_norm_stats_file, 'w'), indent=2)
if not os.path.exists(new_norm_stats_file_with_interpolation):
    print(f'writing {new_norm_stats_file_with_interpolation}')
    json.dump(new_norm_stats, open(new_norm_stats_file_with_interpolation, 'w'), indent=2)
if not os.path.exists(new_norm_stats_file_with_interpolation_longer_act_horizon):
    print(f'writing {new_norm_stats_file_with_interpolation_longer_act_horizon}')
    json.dump(new_norm_stats_longer, open(new_norm_stats_file_with_interpolation_longer_act_horizon, 'w'), indent=2)
