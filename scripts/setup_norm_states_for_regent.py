import json 
import os

norm_stats_basic_file = "assets/droid_basic/norm_stats_simple.json"
new_norm_stats_file = "assets/pi0_fast_droid_regent/droid/norm_stats.json"
os.makedirs("assets/pi0_fast_droid_regent/droid", exist_ok=True)
new_norm_stats_file_with_interpolation = "assets/pi0_fast_droid_regent_with_interpolation/droid/norm_stats.json"
os.makedirs("assets/pi0_fast_droid_regent_with_interpolation/droid", exist_ok=True)

norm_stats_basic = json.load(open(norm_stats_basic_file, 'r'))
num_retrieved = 5

new_norm_stats = {"norm_stats": {}}
for key in norm_stats_basic["norm_stats"]:
    for i in range(num_retrieved):
        prefix = f"retrieved_{i}_"
        new_norm_stats["norm_stats"][f"{prefix}{key}"] = norm_stats_basic["norm_stats"][key]
    prefix = f"query_"
    new_norm_stats["norm_stats"][f"{prefix}{key}"] = norm_stats_basic["norm_stats"][key]

json.dump(new_norm_stats, open(new_norm_stats_file, 'w'), indent=2)
json.dump(new_norm_stats, open(new_norm_stats_file_with_interpolation, 'w'), indent=2)

