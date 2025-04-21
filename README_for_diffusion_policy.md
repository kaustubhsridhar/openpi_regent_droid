Below details distilled from [this google doc](https://docs.google.com/document/d/17IEyInEPRZ184yZ2_4oBcog7rScNPlQy2vQLPV5IMms/edit?tab=t.0)

All passwords are `robotlearning`

# ON FRANKA LAPTOP

Collect data in a folder like
```bash
# dataset_dir/
# ├── demo_0_taken_at_2025-03-04_00-17-49
# ├── demo_1_taken_at_2025-03-04_00-18-56
# ├── ...
```

Convert to mp4s via following. You have to do this in the eva_ksridhar folder on the franka laptop.
```bash
python svo_to_mp4.py <path_to_dataset_dir>
```

Helpers to move data to franka laptop, run above, and move back processed data to here (ivy):
```bash
for FOL in 2025-03-14_move_the_idli_plate_to_the_right 2025-03-17_pick_up_the_poke_ball_and_put_it_in_the_tray 2025-03-27_move_the_squeegee_to_the_right_and_try_to_drag_it
do
    rsync -avzP -e 'ssh' --exclude='*.npz' /home/ksridhar/openpi_regent_droid/regent_droid_preprocessing/collected_demos/${FOL} franka@10.102.204.231:~/eva_ksridhar/data/
done

conda activate eva_ksridhar
python scripts/svo_to_mp4.py ~/eva_ksridhar/data/2025-03-14_move_the_idli_plate_to_the_right 
python scripts/svo_to_mp4.py ~/eva_ksridhar/data/2025-03-17_pick_up_the_poke_ball_and_put_it_in_the_tray 
python scripts/svo_to_mp4.py ~/eva_ksridhar/data/2025-03-27_move_the_squeegee_to_the_right_and_try_to_drag_it

for FOL in 2025-03-14_move_the_idli_plate_to_the_right 2025-03-17_pick_up_the_poke_ball_and_put_it_in_the_tray 2025-03-27_move_the_squeegee_to_the_right_and_try_to_drag_it
do
    rsync -avzP -e 'ssh' franka@10.102.204.231:~/eva_ksridhar/data/${FOL} ~/openpi_regent_droid/diffusion_policy_droid/raw_datasets/
done
```

# ON EXX
move raw datasets to exx (after processing on franka laptop and transfered to ivy as mentioned above)
```bash
rsync -avzP -e 'ssh' /home/ksridhar/openpi_regent_droid/diffusion_policy_droid/raw_datasets/* exx@158.130.52.14:/data3/Projects_archive/orig_droid_learning/orig_droid_learning/dataset/
```

build rls datasets
```bash
conda activate rlds_env

cd /data3/Projects_archive/orig_droid_learning/orig_droid_learning/droid_dataset_builder/droid

# Update the DATA_PATH and LANGUAGE_INSTRUCTION in droid.py

tfds build --overwrite

mv /home/exx/tensorflow_datasets/droid/1.0.0 /data3/Projects_archive/orig_droid_learning/orig_droid_learning/dataset/squeegee
```

train diffusion policy
```bash
conda activate base && conda activate base && conda activate droid_learning_orig && cd /data3/Projects_archive/orig_droid_learning/orig_droid_learning/droid_policy_learning

python robomimic/scripts/config_gen/droid_runs_language_conditioned_rlds_3cams.py --wandb_proj_name droid

# The above will printout something; just run that! with CUDA_VISIBLE_DEVICES=1 before it
```

Transfer checkpoint from exx to here...
```bash
rsync -avzP -e 'ssh' exx@158.130.52.14:/data3/Projects_archive/orig_droid_learning/orig_droid_learning/droid_policy_learning/experiment_log/droid_idliplate/im/diffusion_policy/04-20-None/bz_32_noise_samples_8_sample_weights_1_dataset_names_droid_cams_2cams_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-lang_visenc_VisualCore_fuser_None/20250420191209/models/model_epoch_50.pth ./diffusion_policy_checkpoints_dont_delete/idliplate_ep50.pth

rsync -avzP -e 'ssh' exx@158.130.52.14:/data3/Projects_archive/orig_droid_learning/orig_droid_learning/droid_policy_learning/experiment_log/droid_pokeball/im/diffusion_policy/04-20-None/bz_32_noise_samples_8_sample_weights_1_dataset_names_droid_cams_2cams_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-lang_visenc_VisualCore_fuser_None/20250420191212/models/model_epoch_50.pth ./diffusion_policy_checkpoints_dont_delete/pokeball_ep50.pth

rsync -avzP -e 'ssh' exx@158.130.52.14:/data3/Projects_archive/orig_droid_learning/orig_droid_learning/droid_policy_learning/experiment_log/droid_squeegee/im/diffusion_policy/04-20-None/bz_32_noise_samples_8_sample_weights_1_dataset_names_droid_cams_2cams_goal_mode_None_truncated_geom_factor_0.3_ldkeys_proprio-lang_visenc_VisualCore_fuser_None/20250420194238/models/model_epoch_50.pth ./diffusion_policy_checkpoints_dont_delete/squeegee_ep50.pth
```

...and then from here to franka laptop.
```bash
rsync -azvP -e 'ssh' ./diffusion_policy_checkpoints_dont_delete/* franka@10.102.204.231:~/droid/ckpt/
```

Copy train logs also from exx to here
```bash
rsync -azvP -e 'ssh' exx@158.130.52.14:/data3/Projects_archive/orig_droid_learning/orig_droid_learning/all_train_logs/* ./logs/diffusion_policy_train_logs/
```

# ON FRANKA LAPTOP
deploy diffusion policy
```bash
conda activate droid-orig
cd droid

./deploy_dp.sh <ckpt name only, without .pth, that is in ckpt/ folder>

or 

python scripts/evaluation/evaluate_policy.py -c <path to your .pth checkpoint file here>
```
