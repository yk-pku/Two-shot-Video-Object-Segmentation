#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python eval_youtube_phase2.py \
  --model saves/phase1.pth   \
  --output ../vos_phase2/phase1_right_davis \
  --mem_every 5 \
  --output_all \
  --reverse \
  --davis
