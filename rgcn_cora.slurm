#!/bin/bash
#SBATCH -o ./rgcn_cora/rgcn_cora.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J rgcn_cora            # 任务名称
#SBATCH --gres=gpu:V100:1               # 申请一张GPUV100
#SBATCH -c 5                # 申请CPU核数
#SBATCH -p sugon
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
if [ ! -a "cora.pickle" ]; then
     CUDA_VISIBLE_DEVICES=0 python prepare_dataset.py -d cora
fi
CUDA_VISIBLE_DEVICES=1 python train.py -d cora --bases 30 --hidden 16 --l2norm 5e-4 --testing --experiment rgcn_cora
