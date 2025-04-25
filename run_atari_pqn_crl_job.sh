#!/bin/bash
#SBATCH --account=rrg-tyrell-ab
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=raymond.chua@mail.mcgill.ca
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Set SEED from argument
SEED=$1

# -- Create timestamped output directory
NOW=$(date "+%Y.%m.%d")
TIME=$(date "+%H%M")
OUT_DIR="/home/chuaraym/scratch/exp_sweep/${NOW}/${TIME}_continual_rl_atari_three_games/${SLURM_JOB_ID}_${SEED}"
mkdir -p "$OUT_DIR"

# -- Set job name (optional, works on some clusters only *before* job starts)
# scontrol update JobName="atari_seed${SEED}" JobId=$SLURM_JOB_ID

# -- Redirect SLURM output (stdout + stderr)
exec > "${OUT_DIR}/slurm-${SLURM_JOB_ID}.out" 2>&1

echo "Running Atari experiment with SEED=${SEED}"
echo "Output directory: ${OUT_DIR}"

# Load required modules
module load StdEnv/2023
module load cuda/12.2
module load cudnn/8.9.5.29
module load glfw/3.3.8
module load mujoco/3.1.6
module load blis
module load scipy-stack

export FLEXIBLAS=blis

# Activate Python environment
source /home/chuaraym/pqn_atari_env311/bin/activate

# Change to project root (so purejaxql is importable)
cd /home/chuaraym/purejaxql/

# Run the experiment
python purejaxql/pqn_sf_task_atari_crl.py +alg=pqn_sf_task_atari_crl SEED=${SEED} SAVE_PATH=${OUT_DIR} alg.SF_DIM=32