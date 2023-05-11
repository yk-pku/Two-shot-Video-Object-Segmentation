import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed

import json
import sys


class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump, is_bl, data_file=None, pseudo_gt_root = None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl

        self.videos = []
        self.frames = {}

        # Semi-supersived
        self.full_frames = {}

        # Phase2_train
        self.pseudo_gt_root = pseudo_gt_root
        if pseudo_gt_root != None:
            self.phase2 = True
            self.phase2_frames = {}
            vid_list = sorted(os.listdir(self.pseudo_gt_root))
            for vid in vid_list:
                frames = sorted(os.listdir(os.path.join(self.pseudo_gt_root, vid)))
                self.phase2_frames[vid] = frames
        else:
            self.phase2 = False
            vid_list = sorted(os.listdir(self.im_root))

        with open(data_file, 'r') as f:
            train_data = json.load(f)
            for vid, v_frames in train_data.items():
                if (pseudo_gt_root != None) and (vid not in vid_list):
                    continue
                self.frames[vid] = v_frames
                self.videos.append(vid)
        for vid in list(self.frames.keys()):
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            self.full_frames[vid] = frames


        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        if self.is_bl:
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
            ])
        else:
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
            ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.ori_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video
        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        # Semi
        semi_info = {}
        semi_info['name'] = video
        full_frames = self.full_frames[video]

        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            # Don't want to bias towards beginning/end
            this_max_jump = min(len(frames), self.max_jump)
            start_idx = np.random.randint(len(frames)-this_max_jump+1)
            f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1
            f1_idx = min(f1_idx, len(frames)-this_max_jump, len(frames)-1)

            f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
            f2_idx = min(f2_idx, len(frames)-this_max_jump//2, len(frames)-1)

            frames_idx = [start_idx, f1_idx, f2_idx]
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]

            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (masks[0]==l).sum()
                    if pixel_sum > 10*10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30*30:
                            good_lables.append(l)
                        elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)

            if len(labels) == 0:
                target_object = -1 # all black if no objects
                has_second_object = False
                trials += 1
            else:
                target_object = np.random.choice(labels)
                has_second_object = (len(labels) > 1)
                if has_second_object:
                    labels = labels[labels!=target_object]
                    second_object = np.random.choice(labels)
                break

        masks = np.stack(masks, 0)
        tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:]
        if has_second_object:
            sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:]
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((3, 384, 384), dtype=np.int)
        cls_gt[tar_masks[:,0] > 0.5] = 1
        cls_gt[sec_masks[:,0] > 0.5] = 2

        data = {
            'rgb': images,        # [3, 3, 384, 384]
            'gt': tar_masks,      # [3, 1, 384, 384]
            'cls_gt': cls_gt,     # [3, 384, 384]
            'sec_gt': sec_masks,  # [3, 1, 384, 384]
            'selector': selector, # [2]
            'info': info,
        }

        if self.phase2:
            phase2_info = {}
            phase2_info['name'] = video
            phase2_frames = self.phase2_frames[video]
            vid_pgt_path = path.join(self.pseudo_gt_root, video)
            phase2_info['vid_pgt_path'] = vid_pgt_path
            phase2_info['vid_path'] = vid_im_path

            trials = 0

            while trials < 5:
                phase2_info['frames'] = [] # Appended with actual frames

                this_max_jump = min(len(phase2_frames), self.max_jump)
                start_idx = np.random.randint(len(phase2_frames)-this_max_jump+1)
                f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1
                f1_idx = min(f1_idx, len(phase2_frames)-this_max_jump, len(phase2_frames)-1)

                f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
                f2_idx = min(f2_idx, len(phase2_frames)-this_max_jump//2, len(phase2_frames)-1)

                frames_idx = [start_idx, f1_idx, f2_idx]

                if np.random.rand() < 0.5:
                    # Reverse time
                    frames_idx = frames_idx[::-1]

                inter_idx = None

                sequence_seed = np.random.randint(2147483647)
                images = []
                masks = []
                ori_images = []
                ori_masks = []
                target_object = None
                for f_idx in frames_idx:
                    jpg_name = phase2_frames[f_idx][:-4] + '.jpg'
                    png_name = phase2_frames[f_idx][:-4] + '.png'
                    phase2_info['frames'].append(jpg_name)

                    reseed(sequence_seed)
                    this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')

                    this_im = self.all_im_dual_transform(this_im)
                    this_im = self.all_im_lone_transform(this_im)
                    reseed(sequence_seed)
                    this_gt = Image.open(path.join(vid_pgt_path, png_name)).convert('P')

                    this_gt = self.all_gt_dual_transform(this_gt)

                    pairwise_seed = np.random.randint(2147483647)
                    reseed(pairwise_seed)
                    this_im = self.pair_im_dual_transform(this_im)
                    this_im = self.pair_im_lone_transform(this_im)
                    reseed(pairwise_seed)
                    this_gt = self.pair_gt_dual_transform(this_gt)

                    this_im = self.final_im_transform(this_im)
                    this_gt = np.array(this_gt)

                    images.append(this_im)
                    masks.append(this_gt)

                images = torch.stack(images, 0)

                labels = np.unique(masks[0])
                # Remove background
                labels = labels[labels!=0]

                if len(labels) == 0:
                    target_object = -1 # all black if no objects
                    has_second_object = False
                    trials += 1
                else:
                    target_object = np.random.choice(labels)
                    has_second_object = (len(labels) > 1)
                    if has_second_object:
                        labels = labels[labels!=target_object]
                        second_object = np.random.choice(labels)
                    break
            phase2_info['target_object'] = target_object
            if has_second_object:
                phase2_info['second_object'] = second_object
            else:
                phase2_info['second_object'] = -1

            masks = np.stack(masks, 0)
            tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:]
            if has_second_object:
                sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:]
                selector = torch.FloatTensor([1, 1])
            else:
                sec_masks = np.zeros_like(tar_masks)
                selector = torch.FloatTensor([1, 0])

            cls_gt = np.zeros((3, 384, 384), dtype=np.int)
            cls_gt[tar_masks[:,0] > 0.5] = 1
            cls_gt[sec_masks[:,0] > 0.5] = 2

            phase2_data = {
                'rgb': images,        # [3, 3, 384, 384]
                'gt': tar_masks,      # [3, 1, 384, 384]
                'cls_gt': cls_gt,     # [3, 384, 384]
                'sec_gt': sec_masks,  # [3, 1, 384, 384]
                'selector': selector, # [2]
                'info': phase2_info,
            }
            return data, phase2_data

        # Semi
        frames = sorted(list(np.random.choice(frames, 2, replace = False)))
        full_frames = sorted(full_frames)
        gt_idx = []

        temp_frames = frames.copy()
        temp_full_frames = full_frames.copy()

        if np.random.rand() < 0.5:
            full_frames.reverse()
            frames.reverse()
        for i, frame in enumerate(full_frames):
            if frame == frames[0]:
                gt_idx.append(i)
            elif frame == frames[1]:
                gt_idx.append(i)
                break
            else:
                continue
        trials = 0
        while trials < 5:
            start_gt_idx = np.random.randint(2)
            start_idx = gt_idx[start_gt_idx]

            # start_idx = gt_idx[start_gt_idx]
            this_max_jump = min(len(full_frames), self.max_jump)
            f1_idx = start_idx + np.random.randint(this_max_jump + 1) + 1
            f1_idx = min(f1_idx, len(full_frames) - 1)
            f2_idx = f1_idx + np.random.randint(this_max_jump + 1) + 1
            f2_idx = min(f2_idx, len(full_frames) - 1)

            gt_mask = []
            semi_frames_idx = [start_idx, f1_idx, f2_idx]
            for idx in semi_frames_idx:
                if idx in gt_idx:
                    gt_mask.append(1)
                else:
                    gt_mask.append(0)

            if np.random.rand() > 0.5: # keep 2 gt frames
                if sum(gt_mask) < 2:
                    if start_gt_idx == 1:
                        semi_frames_idx = [gt_idx[0], start_idx, f1_idx]
                        gt_mask = [1, 1, 0]
                    else:
                        for i in range(1, 3):
                            if semi_frames_idx[i] > gt_idx[1]:
                                break
                        if i == 1:
                            semi_frames_idx = [start_idx, gt_idx[1], f1_idx]
                            gt_mask = [1, 1, 0]
                        else:
                            semi_frames_idx = [start_idx, f1_idx, gt_idx[1]]
                            gt_mask = [1, 0, 1]
            gt_mask = torch.FloatTensor(gt_mask)

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            semi_info['frames'] = []
            target_object = None
            for f_idx in semi_frames_idx:
                jpg_name = full_frames[f_idx][:-4] + '.jpg'
                png_name = full_frames[f_idx][:-4] + '.png'
                semi_info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                # print(f'this_im shape is {np.array(this_im).shape}') # (480, 853, 3)
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)
                # print(f'after, this_im shape is {np.array(this_im).shape}') # (384, 384, 3)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]

            if len(labels) == 0:
                target_object = -1 # all black if no objects
                has_second_object = False
                trials += 1
            else:
                target_object = np.random.choice(labels)
                has_second_object = (len(labels) > 1)
                if has_second_object:
                    labels = labels[labels!=target_object]
                    second_object = np.random.choice(labels)
                break

        masks = np.stack(masks, 0)
        tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:]
        if has_second_object:
            sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:]
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((3, 384, 384), dtype=np.int)
        cls_gt[tar_masks[:,0] > 0.5] = 1
        cls_gt[sec_masks[:,0] > 0.5] = 2

        semi_data = {
            'rgb': images,        # [3, 3, 384, 384]
            'gt': tar_masks,      # [3, 1, 384, 384]
            'cls_gt': cls_gt,     # [3, 384, 384]
            'sec_gt': sec_masks,  # [3, 1, 384, 384]
            'selector': selector, # [2]
            'info': semi_info,
            'gt_mask': gt_mask  # [3]
        }

        return data, semi_data

    def aug_frame(self, frame_name, im_path, gt_path, sequence_seed):
        jpg_name = frame_name + '.jpg'
        png_name = frame_name + '.png'

        reseed(sequence_seed)
        this_im = Image.open(path.join(im_path, jpg_name)).convert('RGB')
        this_im = self.all_im_dual_transform(this_im)
        this_im = self.all_im_lone_transform(this_im)
        reseed(sequence_seed)
        this_gt = Image.open(path.join(gt_path, png_name)).convert('P')
        this_gt = self.all_gt_dual_transform(this_gt)

        pairwise_seed = np.random.randint(2147483647)
        reseed(pairwise_seed)
        this_im = self.pair_im_dual_transform(this_im)
        this_im = self.pair_im_lone_transform(this_im)
        reseed(pairwise_seed)
        this_gt = self.pair_gt_dual_transform(this_gt)

        this_im = self.final_im_transform(this_im)
        this_gt = np.array(this_gt)

        return this_im, this_gt

    def __len__(self):
        return len(self.videos)
