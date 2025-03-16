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

new_norm_stats = {"norm_stats": {}}
new_norm_stats_longer = {"norm_stats": {}}
for key in norm_stats_basic["norm_stats"]:
    for i in range(num_retrieved):
        prefix = f"retrieved_{i}_"
        new_norm_stats["norm_stats"][f"{prefix}{key}"] = norm_stats_basic["norm_stats"][key]
        new_norm_stats_longer["norm_stats"][f"{prefix}{key}"] = norm_stats_basic["norm_stats"][key]
    prefix = f"query_"
    new_norm_stats["norm_stats"][f"{prefix}{key}"] = norm_stats_basic["norm_stats"][key]
    new_norm_stats_longer["norm_stats"][f"{prefix}{key}"] = norm_stats_basic["norm_stats"][key]


if not os.path.exists(new_norm_stats_file):
    print(f'writing {new_norm_stats_file}')
    json.dump(new_norm_stats, open(new_norm_stats_file, 'w'), indent=2)
if not os.path.exists(new_norm_stats_file_with_interpolation):
    print(f'writing {new_norm_stats_file_with_interpolation}')
    json.dump(new_norm_stats, open(new_norm_stats_file_with_interpolation, 'w'), indent=2)
if not os.path.exists(new_norm_stats_file_with_interpolation_longer_act_horizon):
    print(f'writing {new_norm_stats_file_with_interpolation_longer_act_horizon}')
    json.dump(new_norm_stats_longer, open(new_norm_stats_file_with_interpolation_longer_act_horizon, 'w'), indent=2)
