#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
  python -m torch.distributed.launch --master_port 9846 --nproc_per_node=2 train.py --stage 3 \
  --id phase2 \
  --load_network saves/stcn_s0.pth\
  --yv_data util/yv_rand_2frames.json \
  --davis_data  util/davis_rand_2frames.json \
  --phase2_yv ../vos_phase2/phase1_merge_480p \
  --phase2_davis ../vos_phase2/phase1_merge_davis \
  --phase2_train 0 --phase2_thres 0.99 --phase2_start_update 70000
