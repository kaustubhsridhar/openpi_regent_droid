# regent + droid + openpi 

## setup
```bash
git clone --recurse-submodules git@github.com:kaustubhsridhar/openpi_regent_droid.git
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
uv pip install tensorflow-datasets tensorflow-cpu autofaiss
uv pip install google-genai openai
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
If you only want to count the number of groupings or just read the json with the language instructions in each grouping, you can do:
```bash
python quick_view_grouping.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 --only_count
```

* I realized that the above groupings hvae too many different tasks in a group. So I first ran the below to save the jsons of the scene_id only groupings with atleast 50 episodes in each grouping:
```bash
python quick_view_grouping.py --chosen_id scene_id --min_num_episodes_in_each_grouping 50 --only_count
```
Then, I ran the following LLM script to create subgroups of the above groupings. This script also has verification to remove hallucinations or inform you of them. It also counts the number of subgroups with atleast 20 demos of which there are 260! Finally, the below script saves a "superdict" of {SceneID_GeminiTaskSummary: {ep_idx: language_instruction, ...}, ...} to the file at `regent_droid_preprocessing/droid_groups/droid_new_superdict_of_subgroups_with_atleast_20_episodes_FULL.json`. Then I copied this json to `regent_droid_preprocessing/droid_groups/droid_new_superdict_of_subgroups_with_atleast_20_episodes.json` and manually edited it and reduced the number of groups to 196 (and also removed some episodes within some groups).
```bash
python create_subgroups_of_groupings_with_llm.py
```


* Check if an object exists in the droid dataset's language annotations
```bash
python check_if_object_in_droid_lang_annotations.py --object_name pinecone
```

* Preprocess the droid dataset groupings, regent-style! (ie by embedding images and doing retrieval to setup training sequences)
```bash
# Embed and retrieve ### this splits the embedding and retrieval into 20 parallel jobs (do 10 groups in the first 19 jobs and 6 groups in the last job for a total of 196 groups)
for TEMPI in {0..19}
do
    CUDA_VISIBLE_DEVICES=9 nohup python -u embed_and_retrieve_within_groupings.py --num_episodes_in_each_grouping 20 --embedding_type embeddings__wrist_image_left --tempi ${TEMPI} &> logs/retrieval_preprocessing/gemini_filtered_206_with_20_episodes_wrist_image_left_${TEMPI}.log &
done
```
If you simply want to embed a single image with pi0 to understand the embedding space, you can do:
```bash
python simple_embed_with_openpi.py
```

You also have to embed and retrieval preprocess some demos we collected in our setup for some seen/training tasks. The method to collect is detailed below but the processing code is here:
```bash
# embed (based on code created for inference time demo preprocessing below)
CUDA_VISIBLE_DEVICES=8 nohup python -u process_collected_demos.py --dir_of_dirs=collected_demos_training &> logs/process_collected_demos/training.txt &

# retrieval_preprocessing
nohup python -u retrieve_within_collected_demo_groups.py &> logs/retrieval_preprocessing/collected_demos_training.log &

```

* Quick view of retrieval preprocessed training sequences with different embedding types
```bash
python quick_view_retrieval_preprocessed_sequences.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 --embedding_type embeddings__wrist_image_left --N 10 --M 5

python quick_view_retrieval_preprocessed_sequences.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 --embedding_type embeddings__exterior_image_1_left --N 10 --M 5

python quick_view_retrieval_preprocessed_sequences.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping 50 --embedding_type both --N 10 --M 5

```

## training
* compute norm stats after processing the collected training data as follows:
```bash
python scripts/setup_norm_states_for_regent.py
```

* train pi0_fast_droid_regent
```bash
# retrieval augmented finetuning
CUDA_VISIBLE_DEVICES=6,9 nohup python -u scripts/train_pi0_fast_regent.py pi0_fast_droid_regent --exp-name=14th_try --overwrite &> logs/log_14.txt &
# adding interpolation below
CUDA_VISIBLE_DEVICES=7,8 nohup python -u scripts/train_pi0_fast_regent.py pi0_fast_droid_regent_with_interpolation --exp-name=14th_try_with_interpolation --overwrite &> logs/log_with_interpolation_14.txt &
# longer action horizon below
CUDA_VISIBLE_DEVICES=6,9 nohup python -u scripts/train_pi0_fast_regent.py pi0_fast_droid_regent_with_interpolation_longer_act_horizon --exp-name=14th_try_with_interpolation_longer_act_horizon --overwrite &> logs/log_with_interpolation_longer_act_horizon_14.txt &
# lower lamda below
CUDA_VISIBLE_DEVICES=0,1 nohup python -u scripts/train_pi0_fast_regent.py pi0_fast_droid_regent_with_interpolation_lamda1 --exp-name=eight_try_with_interpolation_lamda1 --overwrite &> logs/log_with_interpolation_8_lamda1.txt &
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
# You can see output at franka_ksridhar/data/success/date

# copy the demos from the franka laptop to the folder here
rsync -avzP -e 'ssh' franka@10.103.129.112:~/franka_ksridhar/data/success/* regent_droid_preprocessing/collected_demos/

rsync -avzP -e 'ssh' franka@10.103.129.112:~/franka_ksridhar/data/success/* regent_droid_preprocessing/collected_demos_training/

rsync -avzP -e 'ssh' franka@10.103.129.112:~/droid_pi0/results/videos/0314/* videos_dont_delete/pi0_0314_move_the_idli_plate_to_the_right_5times/

rsync -avzP -e 'ssh' franka@10.103.129.112:~/droid_pi0/results_rnp/videos/0314/* videos_dont_delete/rnp_0314_move_the_idli_plate_to_the_right_5times/

rsync -avzP -e 'ssh' franka@10.103.129.112:~/droid_pi0/results_regent/videos/0314/* videos_dont_delete/regent_0314_move_the_idli_plate_to_the_right_5times/
```

* Collect more training demos and transfer as follows:
```bash
rsync -avzP -e 'ssh' franka@10.103.129.112:~/franka_ksridhar/data/success/2025-03-10* regent_droid_preprocessing/collected_demos_training/
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
CUDA_VISIBLE_DEVICES=9 nohup python -u process_collected_demos.py --dir_of_dirs=collected_demos &> logs/process_collected_demos/inference.txt &

# After running the above command, you will see a new file in each demo directory as follows:
# │   │   ├── demo_0_taken_at_2025-03-04_00-17-49
# │   │   │   ├── processed_demo.npz
```

* example inference
```bash
CUDA_VISIBLE_DEVICES=9 python -u scripts/test_pi0_fast_regent.py
CUDA_VISIBLE_DEVICES=9 python -u scripts/test_pi0_fast_regent_longer_act_horizon.py
CUDA_VISIBLE_DEVICES=9 python -u scripts/test_pi0_fast_regent_no_interpolation.py
CUDA_VISIBLE_DEVICES=9 python -u scripts/test_retrieve_and_play.py
```

* run pi0 baseline on the robot
```bash
# Run the server on ivy
CUDA_VISIBLE_DEVICES=9 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=s3://openpi-assets/checkpoints/pi0_fast_droid

# If you want to run the server on the workstation in the lab
cd ~/Projects/openpi-main/ 
./run_pi0_exx.sh

# Run the client on the franka robot
# Terminal 1:
startserver
# Terminal 2:
cd ~/droid_pi0/
conda activate droid_pi0
python3 scripts/main.py --remote_host=158.130.55.26 --remote_port=8000 --external_camera="left"

# change the ip if you are running on the workstation in the lab
python3 scripts/main.py --remote_host=158.130.52.14 --remote_port=8000 --external_camera="left"
```

* run regent inference on the robot
```bash
# Run the server on ivy for regent with interpolation
CUDA_VISIBLE_DEVICES=9 uv run scripts/serve_policy_regent.py policy:checkpoint --policy.config=pi0_fast_droid_regent --policy.dir=checkpoints/pi0_fast_droid_regent/14th_try/3000 --policy.demos_dir=regent_droid_preprocessing/collected_demos/2025-03-14_move_the_idli_plate_to_the_right

# (Alternatively) Run the server on ivy for regent with interpolation
CUDA_VISIBLE_DEVICES=9 uv run scripts/serve_policy_regent.py policy:checkpoint --policy.config=pi0_fast_droid_regent_with_interpolation --policy.dir=checkpoints/pi0_fast_droid_regent_with_interpolation/14th_try_with_interpolation/3000 --policy.demos_dir=regent_droid_preprocessing/collected_demos/2025-03-14_move_the_idli_plate_to_the_right

# (alternatively) Run the server on ivy for regent with interpolation and longer action horizon
CUDA_VISIBLE_DEVICES=9 uv run scripts/serve_policy_regent.py policy:checkpoint --policy.config=pi0_fast_droid_regent_with_interpolation_longer_act_horizon --policy.dir=checkpoints/pi0_fast_droid_regent_with_interpolation_longer_act_horizon/14th_try_with_interpolation_longer_act_horizon/3000 --policy.demos_dir=regent_droid_preprocessing/collected_demos/2025-03-14_move_the_idli_plate_to_the_right

# (Alternatively) Run the server on ivy for retrieve and play
CUDA_VISIBLE_DEVICES=9 uv run scripts/serve_policy_retrieve_and_play.py policy:checkpoint --policy.demos_dir=regent_droid_preprocessing/collected_demos/2025-03-14_move_the_idli_plate_to_the_right

# Run the client on the franka robot
# Terminal 1:
startserver
# Terminal 2:
cd ~/droid_pi0/
conda activate droid_pi0
python3 scripts/main_regent.py --remote_host=158.130.55.26 --remote_port=8000 --external_camera="left" 
# you can get your host computer's public ip via `curl -4 ifconfig.me`
```

