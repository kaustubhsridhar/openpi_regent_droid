import numpy as np
from collections import defaultdict
import json
from openpi.policies.utils import myprint, embed, load_dinov2, embed_with_batches, EMBED_DIM
import os
import argparse
from quick_view_grouping import get_ep_idx_to_info, get_chosen_id_to_ep_idxs
from autofaiss import build_index
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # This prevents JAX from preallocating most of the GPU memory.

def get_all_fol_names(ds_name, args):
	ds_fol = f"{ds_name}_broken_up"
	ds_emb_fol = f"{ds_name}_broken_up_embeddings/chosenID{args.chosen_id}_numepisodes{args.num_episodes_in_each_grouping}"
	os.makedirs(f"{ds_name}_broken_up_embeddings", exist_ok=True)
	os.makedirs(ds_emb_fol, exist_ok=True)
	indices_fol = f"{ds_name}_broken_up_indices_and_distances/chosenID{args.chosen_id}_numepisodes{args.num_episodes_in_each_grouping}_embtype{args.embedding_type}_knnk{args.knn_k}"
	os.makedirs(f"{ds_name}_broken_up_indices_and_distances", exist_ok=True)
	os.makedirs(indices_fol, exist_ok=True)
	return ds_fol, ds_emb_fol, indices_fol

def embed_episodes(chosen_id_to_ep_idxs, ds_fol, ds_emb_fol, num_episodes_to_retrieve_from):
	# init setup
	num_groupings = len(chosen_id_to_ep_idxs)

	# embedding model
	dinov2 = load_dinov2()
	
	# main loop
	for chosen_id_count, (chosen_id, ep_idxs) in enumerate(list(chosen_id_to_ep_idxs.items())[10*args.tempi:10*(args.tempi+1)]):
		# cap ep_idxs
		assert len(ep_idxs) == num_episodes_to_retrieve_from
		
		for ep_count, ep_idx in enumerate(ep_idxs):
			if not os.path.exists(f"{ds_emb_fol}/episode_{ep_idx}_embeddings__exterior_image_1_left.npy"):
				# embed the three videos in the episode
				# read
				steps = np.load(f"{ds_fol}/episode_{ep_idx}.npz")
				observation__exterior_image_1_left = steps["observation__exterior_image_1_left"] # (num_steps, 180, 320, 3)
				observation__exterior_image_2_left = steps["observation__exterior_image_2_left"] # (num_steps, 180, 320, 3)
				observation__wrist_image_left = steps["observation__wrist_image_left"] # (num_steps, 180, 320, 3)
				num_steps = len(observation__exterior_image_1_left)
				assert observation__exterior_image_1_left.dtype == observation__exterior_image_2_left.dtype == observation__wrist_image_left.dtype == np.uint8
				# embed
				embeddings__exterior_image_1_left = embed_with_batches(observation__exterior_image_1_left, dinov2, batch_size=1024) # (num_steps, EMBED_DIM)
				embeddings__exterior_image_2_left = embed_with_batches(observation__exterior_image_2_left, dinov2, batch_size=1024) # (num_steps, EMBED_DIM)
				embeddings__wrist_image_left = embed_with_batches(observation__wrist_image_left, dinov2, batch_size=1024) # (num_steps, EMBED_DIM)
				assert embeddings__exterior_image_1_left.shape[0] == embeddings__exterior_image_2_left.shape[0] == embeddings__wrist_image_left.shape[0] == num_steps
				# save
				np.save(f"{ds_emb_fol}/episode_{ep_idx}_embeddings__exterior_image_1_left.npy", embeddings__exterior_image_1_left)
				np.save(f"{ds_emb_fol}/episode_{ep_idx}_embeddings__exterior_image_2_left.npy", embeddings__exterior_image_2_left)
				np.save(f"{ds_emb_fol}/episode_{ep_idx}_embeddings__wrist_image_left.npy", embeddings__wrist_image_left)
				myprint(f'[embed_episodes] embedded episode {ep_idx} with {num_steps} steps [episode count {ep_count}/{len(ep_idxs)}]')
			else:
				myprint(f'[embed_episodes] skipping episode {ep_idx} [episode count {ep_count}/{len(ep_idxs)}]')
		myprint(f'[embed_episodes] finished embedding all episodes for {chosen_id} [chosen_id count {chosen_id_count}/{num_groupings}]')
	myprint(f'[embed_episodes] done!')

def retrieval_preprocessing(chosen_id_to_ep_idxs, ds_emb_fol, indices_fol, num_episodes_to_retrieve_from, nb_cores_autofaiss, knn_k, embedding_type):
	myprint(f'[retrieval_preprocessing] starting retrieval preprocessing for {embedding_type}')
	
	# init setup
	num_groupings = len(chosen_id_to_ep_idxs)
	all_embedding_types = ["embeddings__exterior_image_1_left", "embeddings__wrist_image_left"]

	# main loop
	for chosen_id_count, (chosen_id, ep_idxs) in enumerate(list(chosen_id_to_ep_idxs.items())[10*args.tempi:10*(args.tempi+1)]):
		# cap ep_idxs
		assert len(ep_idxs) <= num_episodes_to_retrieve_from

		# collect all embeddings and indices
		all_embeddings = []
		all_embeddings_map = {}
		all_indices = []
		for ep_count, ep_idx in enumerate(ep_idxs):
			if embedding_type in all_embedding_types:
				ep_embeddings = np.load(f"{ds_emb_fol}/episode_{ep_idx}_{embedding_type}.npy")
				all_embeddings.append(ep_embeddings)
				all_embeddings_map[ep_idx] = ep_embeddings
			elif embedding_type == "both":
				ep_embeddings = np.concatenate([np.load(f"{ds_emb_fol}/episode_{ep_idx}_{item}.npy") for item in all_embedding_types], axis=1)
				all_embeddings.append(ep_embeddings)
				all_embeddings_map[ep_idx] = ep_embeddings
			else:
				raise ValueError(f'{embedding_type=} is not in {all_embedding_types} and not "both"')
			num_steps = len(ep_embeddings)
			all_indices.extend([[ep_idx, step_idx] for step_idx in range(num_steps)])
		all_embeddings = np.concatenate(all_embeddings, axis=0)
		all_indices = np.array(all_indices)
		embedding_dim = all_embeddings.shape[1]
		num_total = len(all_embeddings)
		myprint(f'[retrieval_preprocessing] concatenated all embeddings and indices for {num_episodes_to_retrieve_from} episodes for {chosen_id} [chosen_id count {chosen_id_count}/{num_groupings}]')
		myprint(f'[retrieval_preprocessing] we have {num_total=} {embedding_dim=}')

		# for each episode, retrieve from all other embeddings
		for ep_count, ep_idx in enumerate(ep_idxs):
			if os.path.exists(f"{indices_fol}/episode_{ep_idx}.npz"):
				myprint(f'[retrieval_preprocessing] skipping episode {ep_idx} [episode count {ep_count}/{num_episodes_to_retrieve_from}]')
				continue

			all_other_episodes_mask = np.array([True if ep_idx_other != ep_idx else False for (ep_idx_other, step_idx_other) in all_indices])
			num_retrieval = np.sum(all_other_episodes_mask)
			this_episode_mask = np.array([True if ep_idx_other == ep_idx else False for (ep_idx_other, step_idx_other) in all_indices])
			num_query = np.sum(this_episode_mask)
			assert num_retrieval + num_query == num_total
			print(f'[retrieval_preprocessing] for episode {ep_idx} [episode count {ep_count}/{num_episodes_to_retrieve_from}], we have {num_retrieval=} {num_query=}')

			# retrieve based on closeness in each type of embedding
			all_other_episodes_embeddings = all_embeddings[all_other_episodes_mask]
			all_other_episodes_indices = all_indices[all_other_episodes_mask]
			this_episode_embeddings = all_embeddings[this_episode_mask]
			this_episode_indices = all_indices[this_episode_mask]
			assert all_other_episodes_embeddings.shape == (num_retrieval, embedding_dim) and all_other_episodes_indices.shape == (num_retrieval, 2), f'{all_other_episodes_embeddings.shape=} {all_other_episodes_indices.shape=}, {num_retrieval=} {embedding_dim=}'
			assert this_episode_embeddings.shape == (num_query, embedding_dim) and this_episode_indices.shape == (num_query, 2)
			assert this_episode_indices.dtype == np.int64 and all_other_episodes_indices.dtype == np.int64

			# create index with all_other_episodes_embeddings
			knn_index, knn_index_infos = build_index(embeddings=all_other_episodes_embeddings, # Note: embeddings have to be float to avoid errors in autofaiss / embedding_reader!
                                            save_on_disk=False,
                                            min_nearest_neighbors_to_retrieve=knn_k + 5, # default: 20
                                            max_index_query_time_ms=10, # default: 10
                                            max_index_memory_usage="25G", # default: "16G"
                                            current_memory_available="50G", # default: "32G"
                                            metric_type='l2',
                                            nb_cores=nb_cores_autofaiss, # default: None # "The number of cores to use, by default will use all cores" as seen in https://criteo.github.io/autofaiss/getting_started/quantization.html#the-build-index-command
                                            )

			# do retrieval from index for this_episode_embeddings
			topk_distances, topk_indices = knn_index.search(this_episode_embeddings, 2 * knn_k)

			# remove -1s and crop to knn_k
			try:
				topk_indices = np.array([[idx for idx in indices if idx != -1][:knn_k] for indices in topk_indices])
			except:
				print(f'---------------------------------------------------Too many -1s from topk_indices ----------------------------------------------------')
				temp_topk_indices = [[idx for idx in indices if idx != -1][:knn_k] for indices in topk_indices]
				print(f'after -1s, min len: {min([len(indices) for indices in temp_topk_indices])}, max len {max([len(indices) for indices in temp_topk_indices])}')
				print(f'-------------------------------------------------------------------------------------------------------------------------------------------')
				print(f'Leaving some -1s in topk_indices and continuing')
				topk_indices = np.array([row+[-1 for _ in range(knn_k-len(row))] for row in temp_topk_indices])
			
			# convert topk_indices to ep_idxs and step_idxs
			retrieved_indices = all_other_episodes_indices[topk_indices]
			assert retrieved_indices.shape == (num_query, knn_k, 2) and retrieved_indices.dtype == np.int64

			# convert to int32
			retrieved_indices = retrieved_indices.astype(np.int32)
			this_episode_indices = this_episode_indices.astype(np.int32)

			# calculate distances between every embedding of retrieved_indices/query_indices and the first retrieved embedding
			myprint(f'[retrieval_preprocessing] calculating distances ...')
			all_distances = []
			for ct in range(num_query):
				retrieved_indices_row = retrieved_indices[ct]
				temp_first_embedding = all_embeddings_map[retrieved_indices_row[0][0]][retrieved_indices_row[0][1]]
				query_ep_idx, query_step_idx = this_episode_indices[ct]
				assert query_ep_idx == ep_idx and query_step_idx == ct
				distances = [0.0] + [np.linalg.norm(all_embeddings_map[e_idx][s_idx] - temp_first_embedding) for e_idx, s_idx in retrieved_indices_row[1:]]
				distances.append(np.linalg.norm(all_embeddings_map[query_ep_idx][query_step_idx] - temp_first_embedding))
				all_distances.append(distances)
			all_distances = np.array(all_distances)
			assert all_distances.shape == (num_query, knn_k + 1), f'{all_distances.shape=} {num_query=} {knn_k=}'

			# save the retrieved indices and this_episode_indices
			np.savez(f"{indices_fol}/episode_{ep_idx}.npz", 
						retrieved_indices=retrieved_indices, 
						query_indices=this_episode_indices,
						distances=all_distances)
			myprint(f'[retrieval_preprocessing] finished and saved retrieval indices for episode {ep_idx} [episode count {ep_count}/{num_episodes_to_retrieve_from}]')
		myprint(f'[retrieval_preprocessing] finished for {chosen_id} [chosen_id count {chosen_id_count}/{num_groupings}]')
	myprint(f'[retrieval_preprocessing] done for {embedding_type=}!')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--chosen_id", type=str, default="scene_id", choices=["location_name", "object_name", "scene_id", "scene_id_and_object_name", "scene_id_and_object_name_and_task_category"])
	parser.add_argument("--num_episodes_in_each_grouping", type=int, default=20)
	parser.add_argument("--nb_cores_autofaiss", type=int, default=8)
	parser.add_argument("--knn_k", type=int, default=100, help="number of nearest neighbors to retrieve")
	parser.add_argument("--embedding_type", type=str, default="embeddings__wrist_image_1_left", choices=["embeddings__exterior_image_1_left", "embeddings__wrist_image_left", "both"]) # "embeddings__exterior_image_2_left", 
	parser.add_argument("--tempi", type=int, default=0, help="can be used to split the retrieval or embedding into parallel jobs; upto you to include in code and use this; otherwise it will have no effect")
	args = parser.parse_args()

	# setup
	ds_name = "droid_new"
	with open(f"droid_groups/droid_new_superdict_of_subgroups_with_atleast_{args.num_episodes_in_each_grouping}_episodes.json", "r") as f:
		superdict = json.load(f)
	chosen_id_to_ep_idxs = {k: [int(subk) for subk in subdict.keys()][:args.num_episodes_in_each_grouping] for k, subdict in superdict.items()}
	ds_fol, ds_emb_fol, indices_fol = get_all_fol_names(ds_name, args)

	# # embed episodes
	embed_episodes(chosen_id_to_ep_idxs=chosen_id_to_ep_idxs, 
					ds_fol=ds_fol, 
					ds_emb_fol=ds_emb_fol, 
					num_episodes_to_retrieve_from=args.num_episodes_in_each_grouping,)
	
	# retrieval preprocessing
	retrieval_preprocessing(chosen_id_to_ep_idxs=chosen_id_to_ep_idxs, 
							ds_emb_fol=ds_emb_fol, 
							indices_fol=indices_fol, 
							num_episodes_to_retrieve_from=args.num_episodes_in_each_grouping, 
							nb_cores_autofaiss=args.nb_cores_autofaiss, 
							knn_k=args.knn_k,
							embedding_type=args.embedding_type)
