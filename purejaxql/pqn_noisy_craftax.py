"""
This script uses BatchRenorm for more effective batch normalization in long training runs.
"""

import copy
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
from safetensors.flax import load_file, save_file

import wandb

from craftax.craftax_env import make_craftax_env_from_name
from purejaxql.utils.craftax_wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
)
from purejaxql.utils.batch_renorm import BatchRenorm
from purejaxql.utils.noisy_net_helpers import NoisyLinear
from jax.scipy.special import logsumexp


class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    norm_type: str = "batch_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, noise_rng: chex.Array, train: bool):
        if self.norm_input:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = BatchRenorm(use_running_average=not train)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: BatchRenorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        for l in range(self.num_layers):
            x = NoisyLinear(features=self.hidden_size)(x, rng=noise_rng)
            x = normalize(x)
            x = nn.relu(x)

        x = NoisyLinear(features=self.action_dim)(x, rng=noise_rng)

        return x


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    q_val: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    basic_env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = basic_env.default_params
    log_env = LogWrapper(basic_env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
        test_env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["TEST_NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["TEST_NUM_ENVS"]),
        )
    else:
        env = BatchEnvWrapper(log_env, num_envs=config["NUM_ENVS"])
        test_env = BatchEnvWrapper(log_env, num_envs=config["TEST_NUM_ENVS"])

    def train(rng):

        original_rng = rng[0]

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=env.action_space(env_params).n,
            hidden_size=config.get("HIDDEN_SIZE", 128),
            num_layers=config.get("NUM_LAYERS", 2),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            noise_rng = jax.random.split(rng, 1)
            print("init_x shape: ", init_x.shape)
            print("noise_rng shape: ", noise_rng.shape)
            network_variables = network.init(rng, init_x, noise_rng=noise_rng, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s, noise_rng = jax.random.split(rng, 4)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    noise_rng=noise_rng,
                    train=False,
                )

                # different eps for each env
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                # eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                # new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
                print("q_vals shape: ", q_vals.shape)
                new_action = jnp.argmax(q_vals)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
                )

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                )
                return (new_obs, new_env_state, rng), (transition, info)

            # step the env
            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            rng, noise_rng = jax.random.split(rng)

            last_q = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                transitions.next_obs[-1],
                noise_rng=noise_rng,
                train=False,
            )
            if config.get("SOFT_ENTROPY", False):
                logits= last_q / config["ENTROPY_COEF"]
                logsumexp_q = logsumexp(logits, axis=-1)
                last_q = config["ENTROPY_COEF"] * logsumexp_q   # to ensure the same scale

                # compute entropy for logging purposes if using soft entropy
                probs = jax.nn.softmax(logits)
                entropy = -jnp.sum(probs * jnp.log(probs + 1e-10))  # avoid log(0)

            else:
                last_q = jnp.max(last_q, axis=-1)

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                )
                lambda_returns = (
                    1 - transition.done
                ) * lambda_returns + transition.done * transition.reward
                next_q = jnp.max(transition.q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            last_q = last_q * (1 - transitions.done[-1])
            lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_util.tree_map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params, rng):

                        if config.get("Q_LAMBDA", False):
                            rng, noise_rng = jax.random.split(rng)
                            q_vals, updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.batch_stats,
                                },
                                minibatch.obs,
                                noise_rng=noise_rng,
                                train=True,
                                mutable=["batch_stats"],
                            )
                        else:
                            rng, noise_rng = jax.random.split(rng)
                            # if not using q_lambda, re-pass the next_obs through the network to compute target
                            all_q_vals, updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.batch_stats,
                                },
                                jnp.concatenate((minibatch.obs, minibatch.next_obs)),
                                noise_rng=noise_rng,
                                train=True,
                                mutable=["batch_stats"],
                            )
                            q_vals, q_next = jnp.split(all_q_vals, 2)
                            q_next = jax.lax.stop_gradient(q_next)
                            q_next = jnp.max(q_next, axis=-1)  # (batch_size,)
                            target = (
                                minibatch.reward
                                + (1 - minibatch.done) * config["GAMMA"] * q_next
                            )

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, (updates, chosen_action_qvals)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params, rng)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    x = x.reshape(
                        -1, *x.shape[2:]
                    )  # num_steps*num_envs (batch_size), ...
                    x = jax.random.permutation(rng, x)  # shuffle the transitions
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )  # num_mini_updates, batch_size/num_mini_updates, ...
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...
                targets = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), lambda_targets
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
                "lr": lr_scheduler(train_state.n_updates),
                "extrinsic rewards": transitions.reward.mean(),
                "entropy": entropy.mean() if config.get("SOFT_ENTROPY", False) else 0,
                "max_probs": jnp.max(probs, axis=-1).mean() if config.get("SOFT_ENTROPY", False) else 0,
            }
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / infos["returned_episode"].sum(),
                infos,
            )
            metrics.update(done_infos)

            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(train_state, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

            # remove achievement metrics if not logging them
            if not config.get("LOG_ACHIEVEMENTS", False):
                metrics = {
                    k: v for k, v in metrics.items() if "achievement" not in k.lower()
                }

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_rng):
                    # log at intervals
                    if (
                        metrics["update_steps"] % config.get("WANDB_LOG_INTERVAL", 128) == 0
                    ):
                        if config.get("WANDB_LOG_ALL_SEEDS", False):
                            metrics.update(
                                {
                                    f"rng{int(original_rng)}/{k}": v
                                    for k, v in metrics.items()
                                }
                            )
                        wandb.log(metrics, step=metrics["update_steps"])

                        # for k, v in metrics.items():
                        #     print(f"{k}: {v}")

                jax.debug.callback(callback, metrics, original_rng)

            runner_state = (train_state, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        def get_test_metrics(train_state, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, _rng, noise_rng = jax.random.split(rng, 3)

                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    noise_rng=noise_rng,
                    train=False,
                )
                # eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                # new_action = jax.vmap(eps_greedy_exploration)(
                #     jax.random.split(_rng, config["TEST_NUM_ENVS"]), q_vals, eps
                # )
                new_action = jnp.argmax(q_vals)
                new_obs, new_env_state, reward, new_done, info = test_env.step(
                    _rng, env_state, new_action, env_params
                )
                return (new_env_state, new_obs, rng), info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.reset(_rng, env_params)

            _, infos = jax.lax.scan(
                _env_step, (env_state, init_obs, _rng), None, config["TEST_NUM_STEPS"]
            )
            # return mean of done infos
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / infos["returned_episode"].sum(),
                infos,
            )
            return done_infos

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)

        rng, _rng = jax.random.split(rng)
        expl_state = env.reset(_rng, env_params)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, expl_state, test_metrics, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def single_run(config):

    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config.get("NAME", f'{config["ALG_NAME"]}_{config["ENV_NAME"]}'),
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"Took {time.time()-t0} seconds to complete.")

    if config.get("SAVE_PATH", None) is not None:
        from purejaxql.utils.save_load import save_params
        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree_util.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "pqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        # return outs

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.001,
                    0.0005,
                    0.0001,
                    0.00005,
                ]
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
