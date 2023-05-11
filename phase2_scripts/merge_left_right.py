import os
import shutil
import json
import numpy as np
from PIL import Image
import sys


# YouTubeVOS
yv_gt_dir = '../../YouTube/phase2/yv_gt_480p'
left_pre_dir = '../../vos_phase2/phase1_left_480p/Annotations'
right_pre_dir  = '../../vos_phase2/phase1_right_480p/Annotations'
leftsecgt_pre_dir = '../../vos_phase2/phase1leftsecgt_480p/Annotations'
rightsecgt_pre_dir = '../../vos_phase2/phase1_rightsecgt_480p/Annotations'
merge_pre_dir  = '../../vos_phase2/phase1_merge_480p/Annotations'


# DAVIS
# yv_gt_dir = '../../DAVIS/2017/phase2/davis_gt'
# left_pre_dir = '../../vos_phase2/phase1_left_davis/Annotations'
# right_pre_dir  = '../../vos_phase2/phase1_right_davis/Annotations'
# leftsecgt_pre_dir = '../../vos_phase2/phase1_leftsecgt_davis/Annotations'
# rightsecgt_pre_dir = '../../vos_phase2/phase1_rightsecgt_davis/Annotations'
# merge_pre_dir  = '../../vos_phase2/phase1_merge_davis/Annotations'


vid_list = os.listdir(yv_gt_dir)
first_frame = dict()
sec_frame = dict()
has_sec = dict()

print('read gts')
for vid in vid_list:
    vid_path = os.path.join(yv_gt_dir, vid)
    frames = sorted(os.listdir(vid_path))
    f_frame = frames[0]
    first_frame[vid] = f_frame
    if len(frames) == 2:
        has_sec[vid] = True
        sec_frame[vid] = frames[1]
    elif len(frames) == 1:
        has_sec[vid] = False
    else:
        print('vid len error')
        os.exit()

os.makedirs(merge_pre_dir, exist_ok = True)

print('merge...')
for vid, f_frame in first_frame.items():
    os.makedirs(os.path.join(merge_pre_dir, vid), exist_ok = True)
    # 0->first_gt, use right_sec_gt pre
    left_first_frames = sorted(os.listdir(os.path.join(rightsecgt_pre_dir, vid)))
    for frame in left_first_frames:
        if frame < f_frame:
            shutil.copy(os.path.join(rightsecgt_pre_dir, vid, frame), os.path.join(merge_pre_dir, vid, frame))
        else:
            break
    # first_gt -> last
    if not has_sec[vid]:
        left_sec_frames = sorted(os.listdir(os.path.join(leftsecgt_pre_dir, vid)))
        for frame in left_sec_frames:
            shutil.copy(os.path.join(leftsecgt_pre_dir, vid, frame), os.path.join(merge_pre_dir, vid, frame))
    else:
        s_frame = sec_frame[vid]
        # first_gt -> sec_gt, use left and right pre
        interval = -1
        left_sec_frames = sorted(os.listdir(os.path.join(left_pre_dir, vid)))
        for frame in left_sec_frames:
            if frame != s_frame:
                interval += 1
            else:
                break
        count_inter = -1
        mid_interval = int(interval / 2)
        for frame in left_sec_frames:
            if frame < s_frame:
                count_inter += 1
                if count_inter <= mid_interval:
                    shutil.copy(os.path.join(left_pre_dir, vid, frame), os.path.join(merge_pre_dir, vid, frame))
                else:
                    shutil.copy(os.path.join(right_pre_dir, vid, frame), os.path.join(merge_pre_dir, vid, frame))
            else:
                break
        # sec_gt -> last
        right_frames = sorted(os.listdir(os.path.join(leftsecgt_pre_dir, vid)))
        for frame in right_frames:
            shutil.copy(os.path.join(leftsecgt_pre_dir, vid, frame), os.path.join(merge_pre_dir, vid, frame))
print('done')
