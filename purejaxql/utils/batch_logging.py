import time

import jax.numpy as jnp
import numpy as np
import wandb

batch_logs = {}
log_times = []


def create_log_dict(info, config):
    to_log = {
        "episode_return": info["returned_episode_returns"],
        "episode_length": info["returned_episode_lengths"],
        # commented out unused keys
        # "num_melee_mobs": info["num_melee_mobs"],
        # "num_ranged_mobs": info["num_ranged_mobs"],
        #'hidden_state': info['hidden_state'],
    }

    if "Craftax" in config["ENV_NAME"]:
        to_log["score"] = info["score"]

        sum_achievements = 0
        for k, v in info.items():
            if "achievements" in k.lower():
                to_log[k] = v
                sum_achievements += v / 100.0

        to_log["achievements"] = sum_achievements

    elif "Tech-Tree" in config["ENV_NAME"]:
        to_log["completed_techs"] = info["completed_techs"]

    if config.get("TRAIN_ICM") or config.get("USE_RND"):
        to_log["intrinsic_reward"] = info["reward_i"]
        to_log["extrinsic_reward"] = info["reward_e"]

        if config.get("TRAIN_ICM"):
            to_log["icm_inverse_loss"] = info["icm_inverse_loss"]
            to_log["icm_forward_loss"] = info["icm_forward_loss"]
        elif config.get("USE_RND"):
            to_log["rnd_loss"] = info["rnd_loss"]

    return to_log

def reset_batch_logs():
    global batch_logs
    global log_times
    log_times = []
    batch_logs = {}

def batch_log(update_step, log, config):
    update_step = int(update_step)
    if update_step not in batch_logs:
        batch_logs[update_step] = []

    batch_logs[update_step].append(log)

    if len(batch_logs[update_step]) == config["NUM_REPEATS"]:
        agg_logs = {}
        for key in batch_logs[update_step][0]:
            agg = []
            if key in ["goal_heatmap"]:
                agg = [batch_logs[update_step][0][key]]
            else:
                for i in range(config["NUM_REPEATS"]):
                    val = batch_logs[update_step][i][key]
                    if not jnp.isnan(val):
                        agg.append(val)

            if len(agg) > 0:
                if key in [
                    "episode_length",
                    "episode_return",
                    "wm_loss",
                    "exploration_bonus",
                    "e_mean",
                    "e_std",
                    "goal_x",
                    "goal_y",
                    "rnd_loss",
                ]:
                    agg_logs[key] = np.mean(agg)
                elif key in ["goal_heatmap"]:
                    agg_logs[key] = wandb.Image(
                        np.array(agg[0]), caption="Goal Heatmap"
                    )
                else:
                    agg_logs[key] = np.array(agg)

        log_times.append(time.time())

        # NOT USED
        # if config["DEBUG"]:
        #     if len(log_times) == 1:
        #         print("Started logging")
        #     elif len(log_times) > 1:
        #         dt = log_times[-1] - log_times[-2]
        #         steps_between_updates = (
        #             config["NUM_ENV_STEPS"] * config["NUM_ENVS"] * config["NUM_REPEATS"]
        #         )
        #         sps = steps_between_updates / dt
        #         agg_logs["sps"] = sps
        #


        wandb.log(agg_logs)


