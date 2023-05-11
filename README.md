# Two-shot Video Object Segmentation
For the first time, we demonstrate the feasibility of two-shot video object segmentation: two labeled frames per video are almost sufficient for training a decent VOS model. 

<div align='center'><img src='./data_show/teaser.png' alt="teaser" height="400px" width='700px'></div>

In this work, we present a simple yet efficient training paradigm to exploit the wealth of information present in unlabeled frames, with only a small amount of labeled data (e.g. 7.3% for YouTube-VOS and 2.9% for DAVIS), our approach still achieves competitive results in contrast to the counterparts trained on full set (2-shot STCN equipped with our approach achieves 85.1%/82.7% on DAVIS 2017/YouTube-VOS 2019, which is -0.1%/-0.0% lower than the STCN trained on full set). 

![overview](./data_show/overview.png)

This work has been accepted by CVPR 2023.

## Installation

This work follows [STCN](https://github.com/hkchengrex/STCN), please install the running environment and prepare datasets according to the corresponding instructions. Besides, we recommend the version of PyTorch >=1.8

## Phase-1

Phase-1 aims to train a STCN model using two labeled frames and their adjacent unlabeled frames, the trained STCN model is used to predict initialized pseudo labels for phase-2.

To run phase-1, you can use:
```
sh run_phase1.sh
```

or using commands:

```
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
```

Note,  you can also use only 



