import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from util.tensor_util import pad_divide_by
from util.tensor_util import unpad
from dataset.range_transform import im_normalization, im_mean
import torch.nn.functional as F

ori_im_transform = transforms.Compose([
    transforms.ToTensor(),
    im_normalization,
])

# Step 1: compare the data
def get_temp_data(vid_path, pgt_path, frames, target_object, second_object):
    ori_images = []
    ori_masks = []
    vid_shape = None
    palette = None
    for frame in frames:
        jpg_name = frame[:-4] + '.jpg'
        png_name = frame[:-4] + '.png'
        this_im = Image.open(os.path.join(vid_path, jpg_name)).convert('RGB')
        ori_images.append(ori_im_transform(this_im))
        this_gt = Image.open(os.path.join(pgt_path, png_name)).convert('P')
        if vid_shape == None:
            vid_shape = np.shape(np.array(this_gt))
        if palette == None:
            palette = Image.open(os.path.join(pgt_path, png_name)).getpalette()
        ori_masks.append(np.array(this_gt))

    ori_images = torch.stack(ori_images, 0)
    ori_masks = np.stack(ori_masks, 0)

    ori_images = ori_images[None, :]
    tar_masks = (ori_masks == target_object).astype(np.float32)
    if second_object != -1:
        sec_masks = (ori_masks == second_object).astype(np.float32)
    else:
        sec_masks = np.zeros_like(tar_masks)

    tar_masks = torch.tensor(tar_masks)
    sec_masks = torch.tensor(sec_masks)
    ori_masks = torch.tensor(ori_masks)
    tar_masks = tar_masks[None, :, None, :]
    sec_masks = sec_masks[None, :, None, :]

    return ori_images.cuda(), ori_masks.cuda(), tar_masks.cuda(), sec_masks.cuda(), vid_shape, palette

def get_ori_prediction(model, imgs, tar_masks, sec_masks):
    selector = torch.FloatTensor([[1, 0]]).cuda()
    with torch.no_grad():
        # Get keys
        k16, kf16_thin, kf16, kf8, kf4 = model('encode_key', imgs)

        # Get values of frame 0
        ref_v1 = model('encode_value', imgs[:,0], kf16[:,0], tar_masks[:,0], sec_masks[:,0])
        ref_v2 = model('encode_value', imgs[:,0], kf16[:,0], sec_masks[:,0], tar_masks[:,0])
        ref_v = torch.stack([ref_v1, ref_v2], 1)

        # Get segment frame 1 base on frame 0
        prev_logits, prev_mask = model('segment', k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1],
                            k16[:,:,0:1], ref_v, selector)

        # Get values of frame 1
        prev_v1 = model('encode_value', imgs[:,1], kf16[:,1], prev_mask[:,0:1], prev_mask[:,1:2])
        prev_v2 = model('encode_value', imgs[:,1], kf16[:,1], prev_mask[:,1:2], prev_mask[:,0:1])
        prev_v = torch.stack([prev_v1, prev_v2], 1)
        values = torch.cat([ref_v, prev_v], 3)

        del ref_v

        # Segment frame2 based frame 0 and 1
        this_logits, this_mask = model('segment', k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2],
                            k16[:,:,0:2], values, selector)

    return  prev_mask.detach(), this_mask.detach()


def update_bank(model, vid_path, pgt_path, frames, target_object, second_object):
    target_object = target_object.item()
    second_object = second_object.item()
    images, ori_masks, tar_masks, sec_masks, vid_shape, palette = get_temp_data(vid_path, pgt_path, frames, target_object, second_object)

    h, w = images.shape[-2:]
    images, pad = pad_divide_by(images, 16)
    tar_masks, _ = pad_divide_by(tar_masks, 16)
    sec_masks, _ = pad_divide_by(sec_masks, 16)

    nh, nw = images.shape[-2:]
    prev_mask, this_mask = get_ori_prediction(model, images, tar_masks, sec_masks)
    frame1_mask = prev_mask[:,0]
    frame2_mask = this_mask[:,0]
    frame1_mask_sec = prev_mask[:,1]
    frame2_mask_sec = this_mask[:,1]
    frame1_mask = frame1_mask[None, :]
    frame2_mask = frame2_mask[None, :]
    frame1_mask_sec = frame1_mask_sec[None, :]
    frame2_mask_sec = frame2_mask_sec[None, :]

    # Sresize the prediction to the original one
    frame1_mask = unpad(frame1_mask, pad)
    frame2_mask = unpad(frame2_mask, pad)
    frame1_mask_sec = unpad(frame1_mask_sec, pad)
    frame2_mask_sec = unpad(frame2_mask_sec, pad)
    frame1_mask = F.interpolate(frame1_mask, vid_shape, mode='bilinear', align_corners=False) # [1, 1, 480, 853]
    frame2_mask = F.interpolate(frame2_mask, vid_shape, mode='bilinear', align_corners=False)
    frame1_mask_sec = F.interpolate(frame1_mask_sec, vid_shape, mode='bilinear', align_corners=False)
    frame2_mask_sec = F.interpolate(frame2_mask_sec, vid_shape, mode='bilinear', align_corners=False)


    # Update ori masks
    ori_masks = ori_masks[1:, :]
    ori_masks[0][frame1_mask[0][0] > 0.99] = target_object
    ori_masks[1][frame2_mask[0][0] > 0.99] = target_object
    ori_masks[0][(1-frame1_mask[0][0]-frame1_mask_sec[0][0]) > 0.99] = 0
    ori_masks[1][(1-frame2_mask[0][0]-frame2_mask_sec[0][0]) > 0.99] = 0

    ori_masks = (ori_masks.detach().cpu().numpy()).astype(np.uint8)

    save_pt_paths = []
    for frame in frames[1:]:
        gt_name = frame[:-4] + '.png'
        save_pt_paths.append(os.path.join(pgt_path, gt_name))

    for i in range(len(save_pt_paths)):
        img_E = Image.fromarray(ori_masks[i])
        img_E.putpalette(palette)
        img_E.save(save_pt_paths[i])

    return
