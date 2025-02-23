import numpy as np
from collections import defaultdict
import json
from utils import myprint, embed, load_policy
import os
import argparse
from quick_view_grouping import get_ep_idx_to_info, get_chosen_id_to_ep_idxs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # This prevents JAX from preallocating most of the GPU memory.

def group_by_chosen_id(chosen_id, total_episodes, min_num_episodes):	
	ep_idx_to_info = get_ep_idx_to_info(total_episodes)
	chosen_id_to_ep_idxs = get_chosen_id_to_ep_idxs(chosen_id, ep_idx_to_info)
	myprint(f'got the info dicts: there are {len(ep_idx_to_info)} groupings\n')

	# create histogram of num_ep_idxs in chosen_id_to_ep_idxs
	chosen_id_to_ep_idxs_with_atleast_min_num_episodes = {chosen_id: ep_idxs for chosen_id, ep_idxs in chosen_id_to_ep_idxs.items() if len(ep_idxs) >= min_num_episodes}
	myprint(f'number of groupings with atleast {min_num_episodes} episodes [in the first {total_episodes} episodes]: {len(chosen_id_to_ep_idxs_with_atleast_min_num_episodes)}\n')
	print('these groupings have chosen_id-->num_episodes as\n\n', {chosen_id: len(ep_idxs) for chosen_id, ep_idxs in chosen_id_to_ep_idxs_with_atleast_min_num_episodes.items()})

	return chosen_id_to_ep_idxs_with_atleast_min_num_episodes

def embed_episodes(chosen_id_to_ep_idxs_with_atleast_min_num_episodes, ds_name, policy):
	ds_fol = f"{ds_name}_broken_up"
	ds_emb_fol = f"{ds_name}_broken_up_embeddings"
	num_groupings = len(chosen_id_to_ep_idxs_with_atleast_min_num_episodes)
	for chosen_id_count, (chosen_id, ep_idxs) in enumerate(chosen_id_to_ep_idxs_with_atleast_min_num_episodes.items()):
		for ep_count, ep_idx in enumerate(ep_idxs):
			if not os.path.exists(f"{ds_emb_fol}/episode_{ep_idx}.npz"):
				# embed the three videos in the episode
				# read
				steps = np.load(f"{ds_fol}/episode_{ep_idx}.npz")
				observation__exterior_image_1_left = steps["observation__exterior_image_1_left"] # (num_steps, 180, 320, 3)
				observation__exterior_image_2_left = steps["observation__exterior_image_2_left"] # (num_steps, 180, 320, 3)
				observation__wrist_image_left = steps["observation__wrist_image_left"] # (num_steps, 180, 320, 3)
				assert observation__exterior_image_1_left.dtype == observation__exterior_image_2_left.dtype == observation__wrist_image_left.dtype == np.uint8
				# embed
				embeddings__exterior_image_1_left = embed(observation__exterior_image_1_left, policy)
				embeddings__exterior_image_2_left = embed(observation__exterior_image_2_left, policy)
				embeddings__wrist_image_left = embed(observation__wrist_image_left, policy)
				# save
				np.savez(f"{ds_emb_fol}/episode_{ep_idx}.npz", 
			 				embeddings__exterior_image_1_left=embeddings__exterior_image_1_left, 
							embeddings__exterior_image_2_left=embeddings__exterior_image_2_left, 
							embeddings__wrist_image_left=embeddings__wrist_image_left)
				myprint(f'embedded episode {ep_idx} [episode count {ep_count}/{len(ep_idxs)}]')
			else:
				myprint(f'skipping episode {ep_idx} [episode count {ep_count}/{len(ep_idxs)}]')
		myprint(f'finished embedding all episodes for {chosen_id} [chosen_id count {chosen_id_count}/{num_groupings}]')
	myprint(f'done embedding!')

def retrieval_preprocessing(chosen_id_to_ep_idxs_with_atleast_min_num_episodes, ds_name):
	ds_emb_fol = f"{ds_name}_broken_up_embeddings"
	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--chosen_id", type=str, default="scene_id_and_object_name", choices=["location_name", "object_name", "scene_id", "scene_id_and_object_name", "scene_id_and_object_name_and_task_category"])
	parser.add_argument("--total_episodes", type=int, default=95658)
	parser.add_argument("--min_num_episodes_in_each_grouping", type=int, default=10)
	args = parser.parse_args()

	# setup
	ds_name = "droid_new"
	chosen_id_to_ep_idxs_with_atleast_min_num_episodes = group_by_chosen_id(args.chosen_id, args.total_episodes, args.min_num_episodes_in_each_grouping)

	# model
	policy = load_policy("pi0_fast_droid")
	
	# embed episodes
	embed_episodes(chosen_id_to_ep_idxs_with_atleast_min_num_episodes, ds_name, policy)
	
	# retrieval preprocessing
	retrieval_preprocessing(chosen_id_to_ep_idxs_with_atleast_min_num_episodes, ds_name)
