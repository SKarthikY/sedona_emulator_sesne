#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 2-00:30
#SBATCH -p iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o ./logs/sedoNNa_small.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/sedoNNa_small.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=yshen99@mit.edu
module load python/3.10.13-fasrc01
source activate torch

python train_small --d_model 128 --nhead 8 --num_layers 4 --learnedPE True --lr 2.5e-4 --weight_decay 0.01 --batch_size 64 --epochs 100