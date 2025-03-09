import numpy as np
from collections import defaultdict
import json
from openpi.policies.utils import myprint, embed, load_dinov2, embed_with_batches
import os
import argparse
from quick_view_grouping import get_ep_idx_to_info, get_chosen_id_to_ep_idxs
from autofaiss import build_index
from embed_and_retrieve_within_groupings import get_all_fol_names
from PIL import Image
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # This prevents JAX from preallocating most of the GPU memory.

def quick_view_retrieval_preprocessed_sequences(ds_fol, ds_emb_fol, indices_fol, embedding_type, knn_k, M, N, total_episodes):
	# randomly sample N files in indices_fol
	indices_fol_files = os.listdir(indices_fol)
	indices_fol_files = [os.path.join(indices_fol, f) for f in indices_fol_files]
	indices_fol_files = np.random.choice(indices_fol_files, size=args.N, replace=False)
	
	# plot all observation types for comparing different embedding types based retrieval results
	observation_types = ["observation__exterior_image_1_left", "observation__wrist_image_left"]

	# create the output folder
	os.makedirs(f"{ds_fol}_quick_view_retrieval_preprocessed_sequences", exist_ok=True)
	os.makedirs(f"{ds_fol}_quick_view_retrieval_preprocessed_sequences/embtype{embedding_type}", exist_ok=True)

	# for each npz file, load the retrieved and query indices
	for file in indices_fol_files:
		myprint(f"Processing {file}")
		npz_file = np.load(file)
		retrieved_indices = npz_file["retrieved_indices"][:, :M, :]
		query_indices = npz_file["query_indices"]
		num_steps = query_indices.shape[0]
		assert retrieved_indices.shape == (num_steps, M, 2) and retrieved_indices.dtype == np.int32
		assert query_indices.shape == (num_steps, 2) and query_indices.dtype == np.int32

		# get all ep_idxs from both the retrieved indices and the one episode index from the query indices
		# get all observation_type images also
		retrieved_ep_idxs = list(np.unique(retrieved_indices[:, :, 0]))
		query_ep_idx = query_indices[0, 0]
		this_ep_idx = int(file.split("_")[-1].split(".")[0])
		assert this_ep_idx == query_ep_idx
		assert this_ep_idx not in retrieved_ep_idxs
		all_images = {ep_idx: np.load(f"{ds_fol}/episode_{ep_idx}.npz") for ep_idx in retrieved_ep_idxs + [query_ep_idx]}

		# create a Nx(M+1) grid of images
		grid_of_images = []
		evenly_spaced_step_idxs = np.linspace(0, num_steps-1, N, dtype=np.int32)
		for i in evenly_spaced_step_idxs:
			for obs_type in observation_types:
				row = []
				q_ep_idx, q_step_idx = query_indices[i, :]
				assert q_step_idx == i
				row.append(all_images[q_ep_idx][obs_type][q_step_idx])
				for j in range(M):
					r_ep_idx, r_step_idx = retrieved_indices[i, j, :]
					row.append(all_images[r_ep_idx][obs_type][r_step_idx])
				grid_of_images.append(np.concatenate(row, axis=1))
		grid_of_images = np.concatenate(grid_of_images, axis=0)
		assert grid_of_images.shape == (N * len(observation_types) * 180, (M+1) * 320, 3) and grid_of_images.dtype == np.uint8

		# save the grid of images
		grid_of_images = Image.fromarray(grid_of_images)
		grid_of_images.save(f"{ds_fol}_quick_view_retrieval_preprocessed_sequences/embtype{embedding_type}/episode_{query_indices[0, 0]}_N{N}_M{M}.png")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--chosen_id", type=str, default="scene_id_and_object_name", choices=["location_name", "object_name", "scene_id", "scene_id_and_object_name", "scene_id_and_object_name_and_task_category"])
	parser.add_argument("--total_episodes", type=int, default=95658)
	parser.add_argument("--min_num_episodes_in_each_grouping", type=int, default=50)
	parser.add_argument("--nb_cores_autofaiss", type=int, default=8)
	parser.add_argument("--knn_k", type=int, default=100, help="number of nearest neighbors to retrieve")
	parser.add_argument("--embedding_type", type=str, default="embeddings__wrist_image_1_left", choices=["embeddings__exterior_image_1_left", "embeddings__wrist_image_left", "both"]) # "embeddings__exterior_image_2_left", 
	parser.add_argument("--N", type=int, default=10, help="number of sequences to sample for the image visualization")
	parser.add_argument("--M", type=int, default=5, help="number of neighbors to sample from each sequence for the image visualization")
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()
	args.num_episodes_to_retrieve_from_in_each_grouping = args.min_num_episodes_in_each_grouping
	assert args.M <= args.knn_k # we are displaying a subset of the retrieved indices

	# setup
	np.random.seed(args.seed)
	ds_name = "droid_new"
	policy_name = "pi0_fast_droid"
	ds_fol, ds_emb_fol, indices_fol = get_all_fol_names(ds_name, args)
	
	quick_view_retrieval_preprocessed_sequences(ds_fol, ds_emb_fol, indices_fol, args.embedding_type, args.knn_k, args.M, args.N, args.total_episodes)

	