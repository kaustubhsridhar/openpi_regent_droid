import os   
import cv2
from PIL import Image

pi0_idli_plate = f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_idli_plate/right_move_the_idli_plate_to_the_right_2025_03_14_17:23:17.mp4.mp4'
regent_idli_plate = f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_idli_plate/ckpt5400longer-0314data-1_move_the_idli_plate_to_the_right_2025_03_16_15:26:52.mp4'
regent_finetune_idli_plate = f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_finetune_idli_plate/finetune-idli-plate-ckpt999-DYNAMIC_move_the_idli_plate_to_the_right_2025_03_24_09:30:37.mp4'

pi0_pokeball = f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_pokeball/right_pick_up_the_pokeball_and_put_it_in_the_tray_2025_04_08_12:43:48.mp4.mp4'
regent_pokeball = f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_pokeball/ckpt5400longer-0317poketraydata-4_pick_up_the_poke_ball_and_put_it_in_the_tray_2025_03_17_13:53:20.mp4'
regent_finetune_pokeball = f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_finetune_pokeball/finetune-pokeball-ckpt999-DYNAMIC_pick_up_the_poke_ball_and_put_it_in_the_tray_2025_03_24_10:55:26.mp4'

pi0_squeegee = f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/pi0_squeegee/right_move_the_squeegee_to_the_right_and_try_to_drag_it_2025_03_27_03:16:21.mp4.mp4'
regent_squeegee = f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_squeegee/ckpt5400longer-0327squeegeedata-6_move_the_squeegee_to_the_right_and_try_to_drag_it_2025_03_27_02:21:50.mp4'
regent_finetune_squeegee = f'/home/ksridhar/openpi_regent_droid/videos_dont_delete/regent_finetune_squeegee/finetune-squeegee-ckpt999-DYNAMIC3_move_the_squeegee_to_the_right_and_try_to_drag_it_2025_03_27_06:47:53.mp4'

pi0_bagel = f''
regent_bagel = f''
regent_finetune_bagel = f''

pi0_lever = f''
regent_lever = f''
regent_finetune_lever = f''

pi0_door = f''
regent_door = f''
regent_finetune_door = f''

