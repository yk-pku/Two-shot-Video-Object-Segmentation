import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model.network import STCN
from model.losses import LossComputer, iou_hooks_mo, iou_hooks_so
from util.log_integrator import Integrator
from util.image_saver import pool_pairs
import sys
import numpy as np
from model.teacher_utils import softmax_mse_loss, softmax_kl_loss, sigmoid_rampup
from model.pseudo_update_utils import *

def update_ema_variables(model, ema_model, global_step, alpha=0.95):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def init_tea(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(0).add_(param.data)


class STCNModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1):
        self.para = para
        self.single_object = para['single_object']
        self.local_rank = local_rank

        self.STCN = nn.parallel.DistributedDataParallel(
            STCN(self.single_object).cuda(),
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        # Mean_Teacher
        self.t_STCN = nn.parallel.DistributedDataParallel(
            STCN(self.single_object).cuda(),
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        self.teacher_init = False

        if self.para['phase2_start_update'] == -1:
            self.phase2_start_update = self.para['phase2_train']
        else:
            self.phase2_start_update = self.para['phase2_start_update']

        if self.para['update_pseudo_bank'] >= 0:
            print('pseudo bank update activated!')

        # Setup logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        if self.single_object:
            self.train_integrator.add_hook(iou_hooks_so)
        else:
            self.train_integrator.add_hook(iou_hooks_mo)
        self.loss_computer = LossComputer(para)

        self.train()
        self.optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, self.STCN.parameters()), lr=para['lr'], weight_decay=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])
        if para['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.report_interval = 100
        self.save_im_interval = 800
        self.save_model_interval = 50000
        if para['debug']:
            self.report_interval = self.save_im_interval = 1

    def do_pass(self, ori_data, semi_data, it = 0, cur_epoch = 0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)
        # Semi
        end_warm = self.para['end_warm']

        if it > end_warm and self.para['semi']:
            data = semi_data
        elif (self.para['phase2_train']) >= 0 and (it >= self.para['phase2_train']):
            data = semi_data
        else:
            data = ori_data
        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)
        Fs = data['rgb']
        Ms = data['gt']

        out = {}
        with torch.cuda.amp.autocast(enabled=self.para['amp']):
            # key features never change, compute once
            k16, kf16_thin, kf16, kf8, kf4 = self.STCN('encode_key', Fs)

            if self.para['use_teacher']:
                with torch.no_grad():
                    t_k16, t_kf16_thin, t_kf16, t_kf8, t_kf4 = self.t_STCN('encode_key', Fs)

            if self.single_object:
                ref_v = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0])

                # Segment frame 1 with frame 0
                prev_logits, prev_mask = self.STCN('segment',
                        k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1],
                        k16[:,:,0:1], ref_v)
                prev_v = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask)

                values = torch.cat([ref_v, prev_v], 2)

                del ref_v

                # Segment frame 2 with frame 0 and 1
                this_logits, this_mask = self.STCN('segment',
                        k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2],
                        k16[:,:,0:2], values)

                out['mask_1'] = prev_mask
                out['mask_2'] = this_mask
                out['logits_1'] = prev_logits
                out['logits_2'] = this_logits
            else:
                # Semi
                if it > end_warm and self.para['semi']:
                    gt_mask = semi_data['gt_mask']

                sec_Ms = data['sec_gt']
                selector = data['selector']

                ref_v1 = self.STCN('encode_value', Fs[:,0], kf16[:,0], Ms[:,0], sec_Ms[:,0])
                ref_v2 = self.STCN('encode_value', Fs[:,0], kf16[:,0], sec_Ms[:,0], Ms[:,0])
                ref_v = torch.stack([ref_v1, ref_v2], 1)

                # Segment frame 1 with frame 0
                prev_logits, prev_mask = self.STCN('segment',
                        k16[:,:,1], kf16_thin[:,1], kf8[:,1], kf4[:,1],
                        k16[:,:,0:1], ref_v, selector)

                prev_v1 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,0:1], prev_mask[:,1:2])
                prev_v2 = self.STCN('encode_value', Fs[:,1], kf16[:,1], prev_mask[:,1:2], prev_mask[:,0:1])
                prev_v = torch.stack([prev_v1, prev_v2], 1)
                values = torch.cat([ref_v, prev_v], 3)

                del ref_v

                if self.para['use_teacher']:
                    with torch.no_grad():
                        t_ref_v1 = self.t_STCN('encode_value', Fs[:,0], t_kf16[:,0], Ms[:,0], sec_Ms[:,0])
                        t_ref_v2 = self.t_STCN('encode_value', Fs[:,0], t_kf16[:,0], sec_Ms[:,0], Ms[:,0])
                        t_ref_v = torch.stack([t_ref_v1, t_ref_v2], 1)
                        t_prev_logits, t_prev_mask = self.t_STCN('segment',
                                t_k16[:,:,1], t_kf16_thin[:,1], t_kf8[:,1], t_kf4[:,1],
                                t_k16[:,:,0:1], t_ref_v, selector)
                        t_prev_v1 = self.t_STCN('encode_value', Fs[:,1], t_kf16[:,1], t_prev_mask[:,0:1], t_prev_mask[:,1:2])
                        t_prev_v2 = self.t_STCN('encode_value', Fs[:,1], t_kf16[:,1], t_prev_mask[:,1:2], t_prev_mask[:,0:1])
                        t_prev_v = torch.stack([t_prev_v1, t_prev_v2], 1)
                        t_values = torch.cat([t_ref_v, t_prev_v], 3)
                        del t_ref_v

                # Segment frame 2 with frame 0 and 1
                this_logits, this_mask = self.STCN('segment',
                        k16[:,:,2], kf16_thin[:,2], kf8[:,2], kf4[:,2],
                        k16[:,:,0:2], values, selector)

                if self.para['use_teacher']:
                    with torch.no_grad():
                        t_this_logits, t_this_mask = self.t_STCN('segment',
                                t_k16[:,:,2], t_kf16_thin[:,2], t_kf8[:,2], t_kf4[:,2],
                                t_k16[:,:,0:2], t_values, selector)
                    t_out = {}
                    t_out['mask_1'] = t_prev_mask[:,0:1]
                    t_out['mask_2'] = t_this_mask[:,0:1]
                    t_out['sec_mask_1'] = t_prev_mask[:,1:2]
                    t_out['sec_mask_2'] = t_this_mask[:,1:2]
                    t_out['logits_1'] = t_prev_logits
                    t_out['logits_2'] = t_this_logits


                out['mask_1'] = prev_mask[:,0:1] # [4, 1, 384, 384]
                out['mask_2'] = this_mask[:,0:1]
                out['sec_mask_1'] = prev_mask[:,1:2] # [4, 1, 384, 384]
                out['sec_mask_2'] = this_mask[:,1:2]

                out['logits_1'] = prev_logits # [4, 3, 384, 384]
                out['logits_2'] = this_logits

                # Semi
                if it > end_warm and self.para['semi']:
                    b, s, _, _ = data['cls_gt'].shape
                    for i in range(b):
                        for j in range(1, s):
                            if gt_mask[i][j] < 0.5:
                                data['cls_gt'][i][j].zero_()

                                if self.para['use_teacher']:
                                    sec_mask_at_ij = t_out['sec_mask_%d'%j][i][0]
                                else:
                                    sec_mask_at_ij = out['sec_mask_%d'%j][i][0]
                                data['cls_gt'][i][j][:] = -1
                                if data['selector'][i][1] > 0.5:
                                    data['cls_gt'][i][j][sec_mask_at_ij >= self.para['semi_thres_upper']] = 2

                                if self.para['use_teacher']:
                                    mask_at_ij = t_out['mask_%d'%j][i][0]
                                else:
                                    mask_at_ij = out['mask_%d'%j][i][0]
                                data['cls_gt'][i][j][mask_at_ij >= self.para['semi_thres_upper']] = 1
                                data['cls_gt'][i][j][(1 - mask_at_ij - sec_mask_at_ij) >= self.para['semi_thres_upper']] = 0

                # Phase2_train
                if (self.para['phase2_train']) >= 0 and (it >= self.para['phase2_train']) and (it >= self.phase2_start_update):
                    b, s, _, _ = data['cls_gt'].shape
                    for i in range(b):
                        update_flag = False
                        vid_path = data['info']['vid_path'][i]
                        vid_pgt_path = data['info']['vid_pgt_path'][i]
                        target_object = data['info']['target_object'][i]
                        second_object = data['info']['second_object'][i]
                        current_frames = np.array(data['info']['frames'])[:, i]
                        for j in range(1, s):
                            if self.para['use_teacher']:
                                sec_mask_at_ij = t_out['sec_mask_%d'%j][i][0]
                            else:
                                sec_mask_at_ij = out['sec_mask_%d'%j][i][0]

                            if data['selector'][i][1] > 0.5:
                                data['cls_gt'][i][j][sec_mask_at_ij >= self.para['phase2_thres']] = 2

                            if self.para['use_teacher']:
                                mask_at_ij = t_out['mask_%d'%j][i][0]
                            else:
                                mask_at_ij = out['mask_%d'%j][i][0]

                            ori_region = torch.sum(data['cls_gt'][i][j] == 1)
                            including = torch.sum(data['cls_gt'][i][j] == 1) == torch.sum((mask_at_ij >= self.para['phase2_thres']) | (data['cls_gt'][i][j] == 1))

                            data['cls_gt'][i][j][mask_at_ij >= self.para['phase2_thres']] = 1
                            data['cls_gt'][i][j][(1 - mask_at_ij - sec_mask_at_ij) >= self.para['phase2_thres']] = 0
                            cur_region = torch.sum(data['cls_gt'][i][j] == 1)

                            if (ori_region != cur_region) and not including:
                                update_flag = True

                        if update_flag and (self.para['update_pseudo_bank'] >= 0) and (it >= self.para['update_pseudo_bank']):
                            self.update_pseudo_label_bank(vid_path, vid_pgt_path, current_frames, target_object, second_object)

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, it)

                # Logging
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.save_im_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                self.logger.log_cv2('train/pairs', pool_pairs(images, size, self.single_object), it)

            if self._is_train:
                if (it) % self.report_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_model_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save(it)

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            if self.para['amp']:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward()
                self.optimizer.step()
            self.scheduler.step()
            if self.para['use_teacher']:
                if not self.teacher_init:
                    init_tea(self.STCN, self.t_STCN)
                    self.teacher_init = True
                    print('init teacher model')
                else:
                    update_ema_variables(self.STCN, self.t_STCN, it, self.para['ema_alpha'])

    def update_pseudo_label_bank(self, vid_path, vid_pgt_path, current_frames, target_object, second_object):
        if self.para['use_teacher']:
            update_bank(self.t_STCN, vid_path, vid_pgt_path, current_frames, target_object, second_object)
        else:
            update_bank(self.STCN, vid_path, vid_pgt_path, current_frames, target_object, second_object)

        return

    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.STCN.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)
        if self.para['use_teacher']:
            t_model_path = self.save_path + ('_%s_tea.pth' % it)
            torch.save(self.t_STCN.module.state_dict(), t_model_path)
            print('Teacher model saved to %s.' % t_model_path)
        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = {
            'it': it,
            'network': self.STCN.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.STCN.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    nn.init.orthogonal_(pads)
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.STCN.module.load_state_dict(src_dict)
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        # Shall be in eval() mode to freeze BN parameters
        self.STCN.eval()
        self.t_STCN.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.STCN.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.STCN.eval()
        return self
