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
# retrieval augmented finetuning
CUDA_VISIBLE_DEVICES=6,9 nohup python -u scripts/train_pi0_fast_regent.py pi0_fast_droid_regent --exp-name=fourth_try_query_loss_only --overwrite &> logs/log_4.txt &
# adding interpolation below
CUDA_VISIBLE_DEVICES=3,4 nohup python -u scripts/train_pi0_fast_regent.py pi0_fast_droid_regent_with_interpolation --exp-name=fourth_try_query_loss_only_with_interpolation --overwrite &> logs/log_with_interpolation_4.txt &
```

## inference 
* Collect demos using franka_ksridhar
```bash
# In the GUI, activate fci and unlock joints. You can access gui on chrome at 172.16.0.2/desk/
# In terminal 1: 
startserver
# In terminal 2: 
startrunnerksridhar
# In terminal 3: 
conda activate droid_ksridhar
cd franka_ksridhar
python scripts/collect_trajectory.py -n 20

# You can see full form of these commands in bashrc 
# The conda env was created by cloning the droid_wliang conda env and then `pip install -e .` in the franka_ksridhar folder
# You can see output at franka_ksridhar/data/date
```

* Process the collected demos as follows
```bash
# The following example structure is expected for the collected demos:
# regent_droid_preprocessing/
# ├── collected_demos/
# │   ├── 2025-03-04/
# │   │   ├── demo_0_taken_at_2025-03-04_00-17-49
# |   │   ├── demo_1_taken_at_2025-03-04_00-18-56
# |   │   ├── ...

# Within each demo directory, we expect the following structure with the traj.h5 file and the recordings' frames:-
# │   │   ├── demo_0_taken_at_2025-03-04_00-17-49
# │   │   │   ├── traj.h5
# │   │   │   ├── recordings/
# |   │   │   │   ├── frames/
# |   │   │   │   │   ├── hand_camera/
# |   │   │   │   │   │   ├── 000.jpg
# |   │   │   │   │   │   ├── 001.jpg
# |   │   │   │   │   │   ├── ...
# |   │   │   │   │   ├── varied_camera_1/
# |   │   │   │   │   │   ├── 000.jpg
# |   │   │   │   │   │   ├── 001.jpg
# |   │   │   │   │   │   ├── ...
# |   │   │   │   │   ├── varied_camera_2/
# |   │   │   │   │   │   ├── 000.jpg
# |   │   │   │   │   │   ├── 001.jpg
# |   │   │   │   │   │   ├── ...

# Process the collected demos; give a few prompts to randomly sample from for each demo (assumtpions: any of these prompts would fit the demo)
cd regent_droid_preprocessing
CUDA_VISIBLE_DEVICES=8 nohup python -u process_collected_demos.py --dir=collected_demos/2025-03-04 --prompts "pick up the pokeball and put it in the bowl" "pick up the pokeball and place it in the bowl" &> logs/process_collected_demos/pokeball_bowl.txt &

# After running the above command, you will see a new file in each demo directory as follows:
# │   │   ├── demo_0_taken_at_2025-03-04_00-17-49
# │   │   │   ├── processed_demo.npz
```

* example inference
```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/test_pi0_fast_regent.py &> log_test.txt &
```

* run pi0 baseline on the robot
```bash
# Run the server on ivy
CUDA_VISIBLE_DEVICES=5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=s3://openpi-assets/checkpoints/pi0_fast_droid

# Run the client on the franka robot
# Terminal 1:
startserver
# Terminal 2:
cd ~/droid_pi0/
conda activate droid_pi0
python3 scripts/main.py --remote_host=158.130.55.26 --remote_port=8000 --external_camera="right"
```

* run regent inference on the robot
```bash
# Run the server on ivy for regent with interpolation
CUDA_VISIBLE_DEVICES=8 uv run scripts/serve_policy_regent.py policy:checkpoint --policy.config=pi0_fast_droid_regent --policy.dir=checkpoints/pi0_fast_droid_regent/fourth_try_query_loss_only/2000 --policy.demos_dir=regent_droid_preprocessing/collected_demos/2025-03-04

# (Alternatively) Run the server on ivy for regent with interpolation
CUDA_VISIBLE_DEVICES=8 uv run scripts/serve_policy_regent.py policy:checkpoint --policy.config=pi0_fast_droid_regent_with_interpolation --policy.dir=checkpoints/pi0_fast_droid_regent_with_interpolation/fourth_try_query_loss_only_with_interpolation/2000 --policy.demos_dir=regent_droid_preprocessing/collected_demos/2025-03-04

# Run the client on the franka robot
# Terminal 1:
startserver
# Terminal 2:
cd ~/droid_pi0/
conda activate droid_pi0
python3 scripts/main_regent.py --remote_host=158.130.55.26 --remote_port=8000 --external_camera="right" 
# you can get your host computer's public ip via `curl -4 ifconfig.me`
```

