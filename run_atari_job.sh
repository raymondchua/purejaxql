#!/bin/bash
#SBATCH --account=rrg-tyrell-ab
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=raymond.chua@mail.mcgill.ca
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Set job name dynamically using environment variable
export SEED=$1

# -- Create timestamped output directory
NOW=$(date "+%Y.%m.%d/%H%M")
OUT_DIR="/home/chuaraym/scratch/exp_sweep/${NOW}_continual_rl_atari_three_games"
mkdir -p "${OUT_DIR}"

# -- Set job name dynamically
JOB_NAME="atari_seed${SEED}"
scontrol update JobName=$JOB_NAME JobId=$SLURM_JOB_ID

# -- Redirect SLURM output (stdout + stderr)
export SLURM_OUTPUT="${OUT_DIR}/slurm-${SLURM_JOB_ID}.out"
exec > "$SLURM_OUTPUT" 2>&1

echo "Running Atari experiment with SEED=${SEED}"
echo "Output directory: ${OUT_DIR}"

module load StdEnv/2023
module load cuda/12.2
module load cudnn/8.9.5.29
module load glfw/3.3.8
module load mujoco/3.1.6
module load blis
module load scipy-stack

export FLEXIBLAS=blis
source /home/chuaraym/pqn_atari_env311/bin/activate

# Change to project root so `purejaxql` is on PYTHONPATH
cd /home/chuaraym/purejaxql/

# Run the experiment
python purejaxql/pqn_atari_crl.py +alg=pqn_atari_crl SEED=${SEED}
