#!/bin/bash
#SBATCH -o ./rgcn_FB15K237/rgcn_FB15K237.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J rgcn_FB            # 任务名称
#SBATCH --gres=gpu:V100:1               # 申请一张GPUV100
#SBATCH -c 5                # 申请CPU核数
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=1 python train.py -d FB15K237 --bases 30 --hidden 16 --l2norm 5e-4 --testing --experiment rgcn_FB15K237
