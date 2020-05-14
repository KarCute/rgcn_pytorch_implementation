#!/bin/bash
#SBATCH -o ./rgcn_WN18RR_4000/rgcn_WN18RR_4000.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J rgcn_W4            # 任务名称
#SBATCH --gres=gpu:V100:1               # 申请一张GPUV100
#SBATCH -c 5                # 申请CPU核数
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
if [ ! -a "WN18RR_4000.pickle" ]; then
     CUDA_VISIBLE_DEVICES=0 python prepare_dataset.py -d WN18RR_4000
fi
CUDA_VISIBLE_DEVICES=1 python train.py -d WN18RR_4000 --bases 30 --hidden 16 --l2norm 5e-4 --testing --experiment rgcn_WN18RR_4000
