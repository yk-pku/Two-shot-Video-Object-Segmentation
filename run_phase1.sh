#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
  python -m torch.distributed.launch --master_port 9846 --nproc_per_node=2 train.py --stage 3 \
  --id phase1 \
  --load_network ckpts/stcn_s0.pth \
  --yv_data util/yv_rand_2frames.json \
  --davis_data  util/davis_rand_2frames.json \
  --semi \
  --semi_thres_upper 0.9 \
  --end_warm 70000 \
  --use_teacher --ema_alpha 0.995
