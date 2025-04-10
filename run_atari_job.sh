#!/bin/bash
#SBATCH --account=rrg-tyrell-ab
#SBATCH --job-name=atari_seed${1}
#SBATCH --output=/home/chuaraym/scratch/exp_sweep/atari_seed${1}_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=raymond.chua@mail.mcgill.ca
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

echo "Running Atari experiment with SEED=${1}"

module load StdEnv/2023
module load cuda/12.2
module load cudnn/8.9.5.29
module load glfw/3.3.8
module load mujoco/3.1.6
module load blis
module load scipy-stack

export FLEXIBLAS=blis
source pqn_atari_env311/bin/activate

python purejaxql/pqn_atari_crl.py +alg=pqn_atari_crl SEED=${1}
