
for TEMPI in 14 #{0..20}
do
    CUDA_VISIBLE_DEVICES=8 nohup python -u embed_and_retrieve_within_groupings.py --num_episodes_in_each_grouping 20 --embedding_type embeddings__wrist_image_left --tempi ${TEMPI} &> logs/retrieval_preprocessing/gemini_filtered_206_with_20_episodes_wrist_image_left_${TEMPI}.log &
done