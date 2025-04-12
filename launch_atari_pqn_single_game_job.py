import subprocess

seeds = [97, 194, 291, 388, 485]

for seed in seeds:
    cmd = ["sbatch", "run_atari_pqn_single_game_job.sh", str(seed)]
    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd)