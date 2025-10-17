#!/bin/sh
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 2-00:30
#SBATCH -p gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o ./logs/sedoNNa_small_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/sedoNNa_small_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=yshen99@mit.edu
module load python/3.10.13-fasrc01
source activate torch

python train_small.py --d_model 256 --nhead 8 --num_layers 6 --learnedPE True --lr 2.5e-4 --weight_decay 0.001 --batch_size 256 --epochs 100