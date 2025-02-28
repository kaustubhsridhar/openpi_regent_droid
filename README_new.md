# regent + droid + openpi 

## setup
```bash
git clone --recurse-submodules git@github.com:kaustubhsridhar/openpi_regent_droid.git
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
uv pip install tensorflow-datasets tensorflow-cpu autofaiss
```

## regent droid preprocessing
* First cd into the folder
```bash
cd regent_droid_preprocessing
```

* Download the droid_new dataset after setting up `gsutil`.  You will need to get permission to access this new version of the droid dataset from the droid authors.
```bash
gsutil -m cp -r gs://xembodiment_data/droid/1.0.1 ./droid_new/
```

* Break up the droid_new dataset into individual episodes
```bash
nohup python -u breakup_droid_dataset.py &> logs/breakup_droid_new.log &
```

* Quickly view what a grouping looks like. A grouping is a list of episodes that have the same scene_id and object_name. Below, you can specify N (number of groupings to sample) and M (number of episodes to sample from each grouping) for the gif visualization.
```bash
python quick_view_grouping.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 --N 10 --M 5
```
If you only want to count the number of groupings, you can do:
```bash
python quick_view_grouping.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 --only_count
```

* Check if an object exists in the droid dataset's language annotations
```bash
python check_if_object_in_droid_lang_annotations.py --object_name pinecone
```

* Preprocess the droid dataset groupings, regent-style! (ie by embedding images and doing retrieval to setup training sequences)
```bash
# Embed (comment out the retrieval part in the code)
CUDA_VISIBLE_DEVICES=${ITEMP} nohup python -u embed_and_retrieve_within_groupings.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 &> logs/embed/scene_id_and_object_name_50_${ITEMP}th_SetOf20.log &

# Retrieve (uncomment the retrieval part in the code and comment out the embedding part)
# for different embedding types, you can do:
CUDA_VISIBLE_DEVICES=0 nohup python -u embed_and_retrieve_within_groupings.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 --embedding_type embeddings__exterior_image_1_left &> logs/retrieval_preprocessing/scene_id_and_object_name_50_exterior_image_1_left.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u embed_and_retrieve_within_groupings.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 --embedding_type embeddings__wrist_image_left &> logs/retrieval_preprocessing/scene_id_and_object_name_50_wrist_image_left.log &

CUDA_VISIBLE_DEVICES=2 nohup python -u embed_and_retrieve_within_groupings.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 --embedding_type both &> logs/retrieval_preprocessing/scene_id_and_object_name_50_both.log &

# Later write a single command with both embedding and retrieval uncommented below
## TODO

```
If you simply want to embed a single image with pi0 to understand the embedding space, you can do:
```bash
python simple_embed_with_openpi.py
```

* Quick view of retrieval preprocessed training sequences with different embedding types
```bash
python quick_view_retrieval_preprocessed_sequences.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 --embedding_type embeddings__wrist_image_left --N 10 --M 5

python quick_view_retrieval_preprocessed_sequences.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 --embedding_type embeddings__exterior_image_1_left --N 10 --M 5

python quick_view_retrieval_preprocessed_sequences.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 --embedding_type both --N 10 --M 5

```

## training
* train pi0_fast_droid_regent
```bash
python scripts/train_pi0_fast_regent.py pi0_fast_droid_regent
```

* train pi0_fast_droid_regent_without_interpolation (regent without dist-weighted-interpolation)
```bash
python scripts/train_pi0_fast_regent.py pi0_fast_droid_regent_without_interpolation
```
