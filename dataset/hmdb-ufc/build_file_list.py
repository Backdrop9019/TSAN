# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import json
import os
import random
from tqdm import tqdm


## open-set
ucf_root = "/data/dataset/ucf101/rawframes"
hmdb_root = "/data/dataset/hmdb51/rawframes"

target_class =['RockClimbingIndoor'
,'Diving'
,'Fencing'
,'GolfSwing'
,'HandstandWalking'
,'SoccerPenalty'
,'PullUps'
,'Punch'
,'PushUps'
,'Biking'
,'HorseRiding'
,'Basketball'
,'Archery'
,'WalkingWithDog']
source_class = ['climb'
,'dive'
,'fencing'
,'golf'
,'handstand'
,'kick_ball'
,'pullup']

# path num_frame groundtruth
train_souce_name = 'hmdb_train_hu'
valid_souce_name = 'hmdb_valid_hu'
train_target_name = 'ucf_train_hu'
valid_target_name = 'ucf_valid_hu'
test_target_name = 'ucf_test_hu'


# with open(osp.join(), 'w') as f:
#     f.writelines()

source_list = [] #hmdb
target_list = [] #ucf

#train_source
label_source_cnt = [0]*8

for label in tqdm(source_class):
    video_list = glob.glob(os.path.join(hmdb_root, label, '*'))
    #num_frame
    #ground_truth3
    info = []
    for video in video_list:
        num_frame =len([name for name in os.listdir(video)])
        ground_truth = source_class.index(label)
        label_source_cnt[ground_truth] += 1
        source_list.append(f'{video} {num_frame} {ground_truth}')

random.shuffle(source_list)
elements = len(source_list)
middle = int(elements * 0.8)
valid_source_list =  source_list[middle:]
train_source_list = source_list[:middle]
#train_target

label_target_cnt = [0]*8
open_set_cnt = [0]*14
for label in tqdm(target_class):
    video_list = glob.glob(os.path.join(ucf_root, label, '*'))
    #video_list /data/dataset/ucf101/rawframes/Archery/v_Swing_g23_c03
    #num_frame
    #ground_truth3
    for video in video_list:
        num_frame =len([name for name in os.listdir(video)])
        ground_truth = target_class.index(label)
        
        if ground_truth >= 7:
            open_set_cnt[ground_truth] +=1
            if open_set_cnt[ground_truth] > 60:
                break
        if ground_truth >= 7:
            ground_truth = 7

 
        target_list.append(f'{video} {num_frame} {ground_truth}')
        label_target_cnt[ground_truth] += 1
        
random.shuffle(target_list)
print(f'train source 개수 : {len(train_source_list)}')
print(f'valid source 개수 : {len(valid_source_list)}')

elements = len(target_list)
train = int(elements * 0.7)
valid = int(elements * 0.1)
target = int(elements * 0.2)
train_target_list = target_list[:train]
valid_target_list = target_list[train:train+valid]
test_target_list = target_list[train+valid:]
print(f'train target 개수 : {len(train_target_list)}')
print(f'valid target 개수 : {len(valid_target_list)}')
print(f'test target 개수 : {len(test_target_list)}')

print(f'label_source_cnt : {label_source_cnt}')
print(f'label_target_cnt : {label_target_cnt}')

with open(f'{train_souce_name}.txt', 'w') as f:
    for item in train_source_list:
        f.write("%s\n" % item)

with open(f'{valid_souce_name}.txt', 'w') as f:
    for item in valid_source_list:
        f.write("%s\n" % item)
with open(f'{train_target_name}.txt', 'w') as f:
    for item in train_target_list:
        f.write("%s\n" % item)
with open(f'{valid_target_name}.txt', 'w') as f:
    for item in valid_target_list:
        f.write("%s\n" % item)
with open(f'{test_target_name}.txt', 'w') as f:
    for item in test_target_list:
        f.write("%s\n" % item)