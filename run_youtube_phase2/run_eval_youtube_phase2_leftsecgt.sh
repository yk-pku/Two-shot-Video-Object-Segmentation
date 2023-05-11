#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python eval_youtube_phase2.py \
  --model saves/phase1.pth   \
  --output ../vos_phase2/phase1_leftsecgt_480p \
  --output_all \
  --res -1 \
  --only_sec_gt \
