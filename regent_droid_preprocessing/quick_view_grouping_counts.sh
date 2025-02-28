mkdir -p logs
mkdir -p logs/quick_view_grouping_counts

for C in 10 20 25 30 40 42 45 50 100
do
    nohup python -u quick_view_grouping.py --chosen_id scene_id_and_object_name --min_num_episodes_in_each_grouping $C --only_count &> logs/quick_view_grouping_counts/${C}_min_num_episodes_in_each_grouping.log &
done