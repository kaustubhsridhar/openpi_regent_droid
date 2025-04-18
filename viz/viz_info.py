import os 

tasks = ['idli_plate', 'pokeball', 'squeegee', 'bagel', 'lever', 'door']
methods = ['pi0', 'regent', 'regent_finetune']


video_paths = {
'pi0_idli_plate' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_idli_plate/right_move_the_idli_plate_to_the_right_2025_03_14_17:17:20.mp4.mp4', 
'regent_idli_plate' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_idli_plate/ckpt5400longer-0314data-1_move_the_idli_plate_to_the_right_2025_03_16_15:26:52.mp4',
'regent_finetune_idli_plate' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_finetune_idli_plate/finetune-idli-plate-ckpt999-DYNAMIC_move_the_idli_plate_to_the_right_2025_03_24_09:30:37.mp4',
'rnp_idli_plate': f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/rnp_idli_plate/rnp_2025_03_19_15:44:29.mp4',
'ablation_5demos_regent_idli_plate': '/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_idli_plate_ablation_5demos/ckpt5400longer-idli-plate-ablation-5demos-8_move_the_idli_plate_to_the_right_2025_03_27_22:32:23.mp4', # behaves like pi0
'ablation_10demos_regent_idli_plate': '/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_idli_plate_ablation_10demos/ckpt5400longer-idli-plate-ablation-10demos-3_move_the_idli_plate_to_the_right_2025_03_27_23:04:03.mp4', # new latent actions
#
'pi0_pokeball' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_pokeball/right_pick_up_the_pokeball_and_put_it_in_the_tray_2025_04_08_12:43:48.mp4.mp4',
'regent_pokeball' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_pokeball/ckpt5400longer-0317poketraydata-4_pick_up_the_poke_ball_and_put_it_in_the_tray_2025_03_17_13:56:25.mp4',
'regent_finetune_pokeball' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_finetune_pokeball/finetune-pokeball-ckpt999-DYNAMIC_pick_up_the_poke_ball_and_put_it_in_the_tray_2025_03_24_10:55:26.mp4',
#
'pi0_squeegee' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_squeegee/right_move_the_squeegee_to_the_right_and_try_to_drag_it_2025_03_27_03:24:15.mp4.mp4', 
'regent_squeegee' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_squeegee/ckpt5400longer-0327squeegeedata-6_move_the_squeegee_to_the_right_and_try_to_drag_it_2025_03_27_02:21:50.mp4',
'regent_finetune_squeegee' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_finetune_squeegee/finetune-squeegee-ckpt999-DYNAMIC3_move_the_squeegee_to_the_right_and_try_to_drag_it_2025_03_27_06:47:53.mp4',
#
'pi0_bagel' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_toaster_bagel/right_pick_up_the_bagel_and_put_it_in_the_toaster_2025_04_07_16:35:40.mp4.mp4',
'regent_bagel' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_toaster_bagel/bagel_8_pick_up_the_bagel_and_put_it_in_the_toaster_2025_04_08_13:39:58.mp4',
'regent_finetune_bagel' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_finetune_toaster_bagel/finetune_bagel_DYNAMIC4_pick_up_the_bagel_and_put_it_in_the_toaster_2025_04_12_00:13:33.mp4',
#
'pi0_lever' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_toaster_lever/right_push_the_lever_on_the_toaster_2025_04_07_16:43:19.mp4.mp4', # aimlessly wanders
'regent_lever' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_toaster_lever/lever_2_push_the_lever_on_the_toaster_2025_04_07_19:16:13.mp4',
'regent_finetune_lever' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_finetune_toaster_lever/finetune_lever_DYNAMIC_push_the_lever_on_the_toaster_2025_04_11_21:46:09.mp4',
#
'pi0_door' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_door_shelf/right_open_the_door_of_the_bottom_shelf_2025_04_11_15:15:00.mp4.mp4', #f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_door_shelf/right_open_the_door_of_the_bottom_locker_2025_04_07_18:12:28.mp4.mp4', # aimlessly wanders
'regent_door' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_door_shelf/door_7_open_the_door_of_the_bottom_shelf_2025_04_11_14:01:09.mp4',
'regent_finetune_door' : f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_finetune_door_shelf/finetune_door_DYNAMIC2_open_the_door_of_the_bottom_shelf_2025_04_11_20:27:24.mp4',
#
# 'pi0_sink-idli-plate': f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_sink-idli-plate/right_move_the_idli_plate_to_the_sink_2025_04_15_00:09:34.mp4.mp4', # moves apple into sink
# above was _old_lang_grounding_issue_only
# below is more adaptation issue (unfamailar grasp or novel motion lack of understanding)
'pi0_sink-idli-plate': f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_sink-idli-plate/right_move_the_idli_plate_to_the_sink_2025_04_15_01:04:16.mp4.mp4',
'regent_sink-idli-plate': f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_sink-idli-plate/sink_idli_8_move_the_idli_plate_to_the_sink_2025_04_15_05:03:56.mp4', # has a bit of time to understand and then does it nicely
#
# 'pi0_sink-squeegee': f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_sink-squeegee/right_use_the_squeegee_to_clean_the_counter_and_push_eve_2025_04_15_01:56:29.mp4.mp4', # goes to plant
# above was _old_lang_grounding_issue_only
# below is more adaptation issue (unfamailar grasp or novel motion lack of understanding)
'pi0_sink-squeegee': f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_sink-squeegee/right_use_the_squeegee_to_clean_the_counter_and_push_eve_2025_04_15_02:02:37.mp4.mp4',
'regent_sink-squeegee': f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_sink-squeegee/sink_squeegee_8_use_the_squeegee_to_clean_the_counter_and_push_eve_2025_04_15_06:20:59.mp4', # you can see the pellets on the top are pushed into the sink
}

chosen_frames = {
'pi0_idli_plate' : [0, 85, 140],
'regent_idli_plate' : [0, 260, 280, 310],
'regent_finetune_idli_plate' : [0, 10, 16, 20, 24, 34, 54],
#
'pi0_pokeball' : [0, 88, 175],
'regent_pokeball' : [0, 20, 36, 66],
'regent_finetune_pokeball' : [15, 16, 19, 24, 25, 58, 94], # removed 0
#
'pi0_squeegee' : [0, 400, 469],
'regent_squeegee' : [0, 78, 94, 110],
'regent_finetune_squeegee' : [10, 17, 19, 25, 54, 63, 81], # can remove 63 or 0
#
'pi0_bagel' : [0, 81, 145],
'regent_bagel' : [0, 50, 134, 136], # can add 64 in there if you want
'regent_finetune_bagel' : [0, 6, 9, 10, 45, 82, 85],
#
'pi0_lever' : [0, 106, 257],
'regent_lever' : [0, 7, 16, 37],
'regent_finetune_lever' : [0, 7, 10, 17, 18, 23, 77],
#
'pi0_door' : [0, 121, 161], #[0, 67, 131],
'regent_door' : [0, 19, 41, 56],
'regent_finetune_door' : [91, 146, 163, 254, 263, 302, 335], # removed 0 # removed 240, 258 and replaced with 254
#
'pi0_sink-idli-plate': [],
'regent_sink-idli-plate': [],
#
'pi0_sink-squeegee': [],
'regent_sink-squeegee': [],
}

