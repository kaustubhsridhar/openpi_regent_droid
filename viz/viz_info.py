import os 

tasks = ['idli_plate', 'pokeball', 'squeegee', 'bagel', 'lever', 'door']
methods = ['pi0', 'regent', 'regent_finetune']

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
'regent_finetune_bagel' : [0, ],
#
'pi0_lever' : [0, 106, 257],
'regent_lever' : [0, 7, 16, 37],
'regent_finetune_lever' : [0, ],
#
'pi0_door' : [0, 67, 131],
'regent_door' : [0, ],
'regent_finetune_door' : [0, ],
}

