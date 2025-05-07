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
from jax.scipy.special import logsumexp
from purejaxql.utils.batch_logging import batch_log, create_log_dict

class SFNetwork(nn.Module):
    action_dim: int
    norm_type: str = "batch_norm"
    norm_input: bool = False
    sf_dim: int = 256
    hidden_size: int = 512
    num_layers: int = 3 # lesser than Q network since we use additional layers to construct SF

    @nn.compact
    def __call__(self, x: jnp.ndarray, task: jnp.ndarray, train: bool):
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
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        rep = nn.Dense(self.sf_dim)(x)
        rep = normalize(rep)
        # rep = nn.tanh(rep)
        rep = nn.relu(rep)
        # basis_features = rep / jnp.linalg.norm(rep, ord=2, axis=-1, keepdims=True)
        basis_features = rep


        task = jax.lax.stop_gradient(task)
        task_normalized = task / jnp.linalg.norm(task, ord=2, axis=-1, keepdims=True)
        rep_task = jnp.concatenate([rep, task_normalized], axis=-1)

        # features for SF
        features_critic_sf = nn.Dense(features=self.sf_dim)(rep_task)
        features_critic_sf = normalize(features_critic_sf)
        features_critic_sf = nn.relu(features_critic_sf)

        # for l in range(self.num_layers):
        #     features_critic_sf = nn.Dense(self.hidden_size)(features_critic_sf)
        #     features_critic_sf = normalize(features_critic_sf)
        #     features_critic_sf = nn.relu(features_critic_sf)

        # SF
        # sf = nn.Dense(features=self.sf_dim * self.action_dim)(features_critic_sf)
        # sf = normalize(sf)
        # sf = nn.relu(sf)

        # create SF for each action
        sf_all = []
        for act in range(self.action_dim):
            sf = nn.Dense(features=self.sf_dim)(features_critic_sf)
            sf = normalize(sf)
            sf = nn.relu(sf)
            sf_all.append(sf)

        sf_action = jnp.stack(sf_all, axis=2)  # (batch_size, sf_dim, action_dim)

        q_1 = jnp.einsum("bi, bij -> bj", task, sf_action).reshape(
            -1, self.action_dim
        )  # (batch_size, action_dim)

        return q_1, basis_features


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

@chex.dataclass
class MultiTrainState:
    network_state: CustomTrainState
    task_state: TrainState

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
            compute_score=config["COMPUTE_SCORE"],
        )
        test_env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["TEST_NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["TEST_NUM_ENVS"]),
            compute_score=config["COMPUTE_SCORE"],
        )
    else:
        env = BatchEnvWrapper(log_env, num_envs=config["NUM_ENVS"], compute_score=config["COMPUTE_SCORE"],)
        test_env = BatchEnvWrapper(log_env, num_envs=config["TEST_NUM_ENVS"], compute_score=config["COMPUTE_SCORE"],)

    env = AddScoreEnvWrapper(env)

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions

    def train(rng):

        original_rng = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        lr_task = config["LR_TASK"]

        # INIT NETWORK AND OPTIMIZER
        network = SFNetwork(
            action_dim=env.action_space(env_params).n,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            sf_dim=config["SF_DIM"],
            hidden_size=config["HIDDEN_SIZE"],
            num_layers=config["NUM_LAYERS"],
        )

        def init_meta(rng, sf_dim, num_env) -> chex.Array:
            _, task_rng_key = jax.random.split(rng)
            task = jax.random.uniform(task_rng_key, shape=(sf_dim,))
            task = task / jnp.linalg.norm(task, ord=2)
            task = jnp.tile(task, (num_env, 1))
            return task

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            init_task = jnp.zeros((1, config["SF_DIM"]))
            network_variables = network.init(rng, init_x, init_task, train=False)
            task_params = {"w": init_meta(rng, config["SF_DIM"], config["NUM_ENVS"])}

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            tx_task = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr_task),
            )

            network_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )

            task_state = TrainState.create(
                apply_fn=network.apply,
                params=task_params,
                tx=tx_task,
            )

            return MultiTrainState(
                network_state=network_state,
                task_state=task_state
            )

        rng, _rng = jax.random.split(rng)
        multi_train_state = create_agent(rng)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            multi_train_state, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                (q_vals, _) = network.apply(
                    {
                        "params": multi_train_state.network_state.params,
                        "batch_stats": multi_train_state.network_state.batch_stats,
                    },
                    last_obs,
                    task=multi_train_state.task_state.params["w"],
                    train=False,
                )

                # different eps for each env
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(multi_train_state.network_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

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

            multi_train_state.network_state = multi_train_state.network_state.replace(
                timesteps=multi_train_state.network_state.timesteps
                          + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            (last_q, _) = network.apply(
                {
                    "params": multi_train_state.network_state.params,
                    "batch_stats": multi_train_state.network_state.batch_stats,
                },
                transitions.next_obs[-1],
                task=multi_train_state.task_state.params["w"],
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
                multi_train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    multi_train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        (all_q_vals, basis_features), updates = network.apply(
                            {
                                "params": params,
                                "batch_stats": multi_train_state.network_state.batch_stats,
                            },
                            jnp.concatenate((minibatch.obs, minibatch.next_obs)),
                            train=True,
                            mutable=["batch_stats"],
                            task=jnp.concatenate((multi_train_state.task_state.params["w"],
                                                  multi_train_state.task_state.params["w"])),
                        )

                        q_vals, q_next = jnp.split(all_q_vals, 2)

                        if not config.get("Q_LAMBDA", False):
                            q_next = jax.lax.stop_gradient(q_next)
                            q_next = jnp.max(q_next, axis=-1)  # (batch_size,)
                            target = (
                                    minibatch.reward
                                    + (1 - minibatch.done) * config["GAMMA"] * q_next
                            )

                            print("concat obs shape: ", jnp.concatenate((minibatch.obs, minibatch.next_obs)).shape)
                            print("concat task shape: ", jnp.concatenate((multi_train_state.task_state.params["w"], multi_train_state.task_state.params["w"])).shape)
                            print("q lambda target shape", target.shape)


                        # prepare basis features for reward prediction
                        basis_features = jnp.split(basis_features, 2)
                        basis_features_next_obs = jax.lax.stop_gradient(basis_features[1])

                        print("basis features next obs shape", basis_features_next_obs.shape)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, (updates, chosen_action_qvals, basis_features_next_obs)

                    def _reward_loss_fn(task_params, basis_features_next_obs):
                        reward = minibatch.reward
                        predicted_reward = jnp.einsum("ij,ij->i", basis_features_next_obs, task_params["w"])
                        loss = 0.5 * jnp.square(predicted_reward - reward).mean()

                        return loss

                    (loss, (updates, qvals, basis_features_next_obs)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(multi_train_state.network_state.params)
                    multi_train_state.network_state = multi_train_state.network_state.apply_gradients(grads=grads)
                    multi_train_state.network_state = multi_train_state.network_state.replace(
                        grad_steps=multi_train_state.network_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )

                    # update task params using reward prediction loss
                    old_task_params = multi_train_state.task_state.params["w"]
                    reward_loss, grads_task = jax.value_and_grad(
                        _reward_loss_fn
                    )(multi_train_state.task_state.params, basis_features_next_obs)
                    multi_train_state.task_state = multi_train_state.task_state.apply_gradients(grads=grads_task)
                    new_task_params = multi_train_state.task_state.params["w"]

                    task_params_diff = jnp.linalg.norm(new_task_params - old_task_params, ord=2, axis=-1)
                    return (multi_train_state, rng), (loss, qvals, reward_loss, task_params_diff)

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

                targets = jnp.reshape(targets, (config["NUM_MINIBATCHES"], config["NUM_ENVS"], -1))

                rng, _rng = jax.random.split(rng)
                (multi_train_state, rng), (loss, qvals, reward_loss, task_params_diff) = jax.lax.scan(
                    _learn_phase, (multi_train_state, rng), (minibatches, targets)
                )

                return (multi_train_state, rng), (loss, qvals, reward_loss, task_params_diff)

            rng, _rng = jax.random.split(rng)
            (multi_train_state, rng), (loss, qvals, reward_loss, task_params_diff) = jax.lax.scan(
                _learn_epoch, (multi_train_state, rng), None, config["NUM_EPOCHS"]
            )

            multi_train_state.network_state = multi_train_state.network_state.replace(n_updates=multi_train_state.network_state.n_updates + 1)
            metrics = {
                "env_step": multi_train_state.network_state.timesteps,
                "update_steps": multi_train_state.network_state.n_updates,
                "grad_steps": multi_train_state.network_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
                "eps": eps_scheduler(multi_train_state.network_state.n_updates),
                "lr": lr_scheduler(multi_train_state.network_state.n_updates) if config.get("LR_LINEAR_DECAY", False) else config["LR"],
                "reward_loss": reward_loss.mean(),
                "task_params_diff": task_params_diff.mean(),
                "extrinsic_rewards": transitions.reward.mean(),
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
                    multi_train_state.network_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(multi_train_state, _rng),
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
                        # wandb.log(metrics, step=metrics["update_steps"])

                        to_log = create_log_dict(metrics, config)
                        metrics.update({k: v for k, v in to_log.items()})
                        batch_log(metrics["update_steps"], metrics, config)

                        for k, v in metrics.items():
                            print(f"{k}: {v}")

                        # to_log = create_log_dict(metrics, config)
                        # batch_log(update_step, to_log, config)
                        #
                        # for k, v in to_log.items():
                        #     print(f"{k}: {v}")





                jax.debug.callback(callback, metrics, original_rng)

            runner_state = (multi_train_state, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        def get_test_metrics(multi_train_state, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)
                q_vals = network.apply(
                    {
                        "params": multi_train_state.network_state.params,
                        "batch_stats": multi_train_state.network_state.batch_stats,
                    },
                    last_obs,
                    task=multi_train_state.task_state.params["w"],
                    train=False,
                )
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(_rng, config["TEST_NUM_ENVS"]), q_vals, eps
                )
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
        test_metrics = get_test_metrics(multi_train_state, _rng)

        rng, _rng = jax.random.split(rng)
        expl_state = env.reset(_rng, env_params)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (multi_train_state, expl_state, test_metrics, _rng)

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
    print(f"Took {time.time() - t0} seconds to complete.")

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
            params = jax.tree_util.tree_map(lambda x: x[i], model_state.network_state.params)
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
