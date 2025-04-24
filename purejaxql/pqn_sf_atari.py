"""
When test_during_training is set to True, an additional number of parallel test environments are used to evaluate the agent during training using greedy actions,
but not for training purposes. Stopping training for evaluation can be very expensive, as an episode in Atari can last for hundreds of thousands of steps.
"""

import copy
import time
import os
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import wandb

import envpool

from purejaxql.utils.atari_wrapper import JaxLogEnvPoolWrapper
from purejaxql.utils.l2_normalize import l2_normalize

class CNN(nn.Module):

    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        return x

class SFNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False
    feature_dim: int = 128
    sf_dim: int = 256
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray, task: jnp.ndarray, train: bool):
        x = jnp.transpose(x, (0, 2, 3, 1))
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0
        x = CNN(norm_type=self.norm_type)(x, train)
        rep = nn.Dense(self.sf_dim)(x)
        # # rep = nn.LayerNorm()(rep)
        # # rep = nn.tanh(rep)
        # # rep = nn.Dense(self.sf_dim)(rep)
        basis_features = rep / jnp.linalg.norm(rep, ord=2, axis=-1, keepdims=True)
        #
        task = jax.lax.stop_gradient(task)
        # task_normalized = task / jnp.linalg.norm(task, ord=2, axis=-1, keepdims=True)
        # rep_task = jnp.concatenate([rep, task_normalized], axis=-1)
        #
        # # features for SF
        # features_critic_sf = nn.Dense(features=self.hidden_dim)(rep)
        # # features_critic_sf = nn.Dense(features=self.feature_dim)(rep_task)
        # features_critic_sf = nn.relu(features_critic_sf)
        # features_critic_sf = nn.Dense(features=self.hidden_dim)(features_critic_sf)
        # features_critic_sf = nn.relu(features_critic_sf)
        #
        # # SF
        # sf = nn.Dense(features=self.sf_dim * self.action_dim)(features_critic_sf)
        # sf_action = jnp.reshape(
        #     sf,
        #     (
        #         -1,
        #         self.sf_dim,
        #         self.action_dim,
        #     ),
        # )  # (batch_size, sf_dim, action_dim)
        #
        # q_1 = jnp.einsum("bi, bij -> bj", task, sf_action).reshape(
        #     -1, self.action_dim
        # )  # (batch_size, action_dim)

        # x = CNN(norm_type=self.norm_type)(x, train)
        sf = nn.Dense(features=10 * self.action_dim)(rep)
        q_1 = nn.Dense(self.action_dim)(sf)

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
    exploration_updates: int = 0
    total_returns: int = 0

@chex.dataclass
class MultiTrainState:
    network_state: CustomTrainState
    task_state: TrainState


def init_meta(rng, sf_dim, num_env) -> chex.Array:
    _, task_rng_key = jax.random.split(rng)
    task = jax.random.uniform(task_rng_key, shape=(sf_dim,))
    task = task / jnp.linalg.norm(task, ord=2)
    task = jnp.tile(task, (num_env, 1))
    return task


def create_agent(rng, config, max_num_actions, observation_space_shape):
    network = SFNetwork(
        action_dim=max_num_actions,
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
        sf_dim=config["SF_DIM"],
        feature_dim=config["FEATURE_DIM"],
    )

    init_x = jnp.zeros((1, *observation_space_shape))
    init_task = jnp.zeros((1, config["SF_DIM"]))
    network_variables = network.init(rng, init_x, init_task, train=False)
    task_params = {"w": init_meta(rng, config["SF_DIM"], config["NUM_ENVS"] + config["TEST_ENVS"])}

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.radam(learning_rate=config["LR"]),
    )

    tx_task = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.radam(learning_rate=config["LR_TASK"]),
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
    ), network


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

    def make_env(num_envs):
        env = envpool.make(
            config["ENV_NAME"],
            env_type="gym",
            num_envs=num_envs,
            seed=config["SEED"],
            full_action_space=config["FULL_ACTION_SPACE"],
            **config["ENV_KWARGS"],
        )
        env.num_envs = num_envs
        env.single_action_space = env.action_space
        env.single_observation_space = env.observation_space
        env.name = config["ENV_NAME"]
        env = JaxLogEnvPoolWrapper(env)
        return env

    total_envs = (
        (config["NUM_ENVS"] + config["TEST_ENVS"])
        if config.get("TEST_DURING_TRAINING", False)
        else config["NUM_ENVS"]
    )
    env = make_env(total_envs)

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

    # here reset must be out of vmap and jit
    init_obs, env_state = env.reset()

    def train(rng, train_state, network):

        original_seed = rng[0]

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

        rng, _rng = jax.random.split(rng)
        train_state.network_state = train_state.network_state.replace(exploration_updates=0)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                (q_vals, _) = network.apply(
                    {
                        "params": train_state.network_state.params,
                        "batch_stats": train_state.network_state.batch_stats,
                    },
                    last_obs,
                    train_state.task_state.params["w"],
                    train=False,
                )

                # different eps for each env
                _rngs = jax.random.split(rng_a, total_envs)
                eps = jnp.full(config["NUM_ENVS"], config["EPS_FINISH"])

                if config.get("TEST_DURING_TRAINING", False):
                    eps = jnp.concatenate((eps, jnp.zeros(config["TEST_ENVS"])))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    env_state, new_action
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

            task_params_target = train_state.task_state.params["w"]

            if config.get("TEST_DURING_TRAINING", False):
                # remove testing envs
                transitions = jax.tree_util.tree_map(
                    lambda x: x[:, : -config["TEST_ENVS"]], transitions
                )
                task_params_target = train_state.task_state.params["w"][: -config["TEST_ENVS"], :]

            train_state.network_state = train_state.network_state.replace(
                timesteps=train_state.network_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            train_state.network_state = train_state.network_state.replace(
                total_returns=train_state.network_state.total_returns + transitions.reward.sum()
            )  # update total returns count

            (last_q,_) = network.apply(
                {
                    "params": train_state.network_state.params,
                    "batch_stats": train_state.network_state.batch_stats,
                },
                transitions.next_obs[-1],
                task_params_target,
                train=False,
            )
            last_q = jnp.max(last_q, axis=-1)

            def _compute_targets(last_q, q_vals, reward, done):
                def _get_target(lambda_returns_and_next_q, rew_q_done):
                    reward, q, done = rew_q_done
                    lambda_returns, next_q = lambda_returns_and_next_q
                    target_bootstrap = reward + config["GAMMA"] * (1 - done) * next_q
                    delta = lambda_returns - next_q
                    lambda_returns = (
                        target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                    )
                    lambda_returns = (1 - done) * lambda_returns + done * reward
                    next_q = jnp.max(q, axis=-1)
                    return (lambda_returns, next_q), lambda_returns

                lambda_returns = reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                last_q = jnp.max(q_vals[-1], axis=-1)
                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_q),
                    jax.tree_util.tree_map(lambda x: x[:-1], (reward, q_vals, done)),
                    reverse=True,
                )
                targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                return targets

            lambda_targets = _compute_targets(
                last_q, transitions.q_val, transitions.reward, transitions.done
            )

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        (q_vals, basis_features), updates = network.apply(
                            {"params": params, "batch_stats": train_state.network_state.batch_stats},
                            minibatch.obs,
                            train_state.task_state.params["w"][: -config["TEST_ENVS"], :],
                            train=True,
                            mutable=["batch_stats"],
                        )  # (batch_size*2, num_actions)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, (updates, chosen_action_qvals, basis_features)

                    def _reward_loss_fn(task_params, basis_features, reward):
                        task_params_train = task_params["w"][:-config["TEST_ENVS"], : ]
                        predicted_reward = jnp.einsum("ij,ij->i", basis_features, task_params_train)
                        loss = 0.5 * jnp.square(predicted_reward - reward).mean()

                        return loss

                    (loss, (updates, qvals, basis_features)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.network_state.params)
                    train_state.network_state = train_state.network_state.apply_gradients(grads=grads)
                    train_state.network_state = train_state.network_state.replace(
                        grad_steps=train_state.network_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )

                    # update task params using reward prediction loss
                    old_task_params = train_state.task_state.params["w"]
                    basis_features = jax.lax.stop_gradient(basis_features)
                    reward_loss, grads_task = jax.value_and_grad(
                        _reward_loss_fn
                    )(train_state.task_state.params, basis_features, minibatch.reward)
                    train_state.task_state = train_state.task_state.apply_gradients(grads=grads_task)
                    new_task_params = train_state.task_state.params["w"]

                    task_params_diff = jnp.linalg.norm(new_task_params - old_task_params, ord=2, axis=-1)

                    return (train_state, rng), (loss, qvals, reward_loss, task_params_diff)

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
                (train_state, rng), (loss, qvals, reward_loss, task_params_diff) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )

                return (train_state, rng), (loss, qvals, reward_loss, task_params_diff)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals, reward_loss, task_params_diff) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state.network_state = train_state.network_state.replace(n_updates=train_state.network_state.n_updates + 1)
            train_state.network_state = train_state.network_state.replace(
                exploration_updates=train_state.network_state.exploration_updates + 1
            )

            if config.get("TEST_DURING_TRAINING", False):
                test_infos = jax.tree_util.tree_map(
                    lambda x: x[:, -config["TEST_ENVS"] :], infos
                )
                infos = jax.tree_util.tree_map(
                    lambda x: x[:, : -config["TEST_ENVS"]], infos
                )
                infos.update({"test/" + k: v for k, v in test_infos.items()})

            metrics = {
                "env_step": train_state.network_state.timesteps,
                "update_steps": train_state.network_state.n_updates,
                "env_frame": train_state.network_state.timesteps
                * env.observation_space.shape[
                    0
                ],  # first dimension of the observation space is number of stacked frames
                "grad_steps": train_state.network_state.grad_steps,
                "td_loss": loss.mean(),
                "reward_loss": reward_loss.mean(),
                "qvals": qvals.mean(),
                "eps": eps_scheduler(train_state.network_state.exploration_updates),
                "lr": lr,
                "exploration_updates": train_state.network_state.exploration_updates,
                "total_returns": train_state.network_state.total_returns,
                "task_params_diff": task_params_diff.mean(),
            }

            metrics.update({k: v.mean() for k, v in infos.items()})
            if config.get("TEST_DURING_TRAINING", False):
                metrics.update({f"test/{k}": v.mean() for k, v in test_infos.items()})

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )

                    for k, v in metrics.items():
                        # check if values are of type numpy.ndarray, if so convert them to float or int using item()
                        if isinstance(v, np.ndarray):
                            metrics[k] = v.item()

                        if metrics["update_steps"] % 10 == 0:
                            print(f"{k}: {v}")
                            # if k == "env_step":
                            #     print(f"{k}: {v}")
                            # if k == "update_steps":
                            #     print(f"{k}: {v}")
                            # if k == "total_returns":
                            #     print(f"{k}: {v}")
                            # if k == "reward_loss":
                            #     print(f"{k}: {v}")
                            # if k == "returned_episode_returns":
                            #     print(f"{k}: {v}")
                            # if k == "rewards":
                            #     print(f"{k}: {v}")

                        # print(f"{k}: {v}")
                        # if k == "env_step":
                        #     print(f"{k}: {v}")
                        #
                        # if k == "update_steps":
                        #     print(f"{k}: {v}")
                        #
                        # if k == "eps":
                        #     print(f"{k}: {v}")

                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        # test metrics not supported yet
        test_metrics = None

        # train
        rng, _rng = jax.random.split(rng)
        expl_state = (init_obs, env_state)
        runner_state = (train_state, expl_state, test_metrics, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics, "train_state": runner_state[0]}

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
        name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    # determine the max number of actions
    max_num_actions = 18  # atari has at most 18 actions
    observation_space_shape = (4, 84, 84)

    rng = jax.random.PRNGKey(config["SEED"])
    rng, rng_agent = jax.random.split(rng)
    agent_train_state, network = create_agent(rng_agent, config, max_num_actions, observation_space_shape)

    t0 = time.time()
    if config["NUM_SEEDS"] > 1:
        raise NotImplementedError("Vmapped seeds not supported yet.")
    else:
        outs = jax.jit(
            lambda rng: make_train(config)(rng, agent_train_state, network)
        )(rng)
    print(f"Took {time.time()-t0} seconds to complete.")

    # save params
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

        # assumes not vmpapped seeds
        params = model_state.params
        save_path = os.path.join(
            save_dir,
            f'{alg_name}_{env_name}_seed{config["SEED"]}.safetensors',
        )
        save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {
        **default_config,
        **default_config["alg"],
    }  # merge the alg config with the main config

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config["alg"][k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])

        if config["NUM_SEEDS"] > 1:
            raise NotImplementedError("Vmapped seeds not supported yet.")
        else:
            outs = jax.jit(make_train(config))(rng)

    sweep_config = {
        "name": f"pqn_atari_{default_config['ENV_NAME']}",
        "method": "bayes",
        "metric": {
            "name": "test_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {"values": [0.0005, 0.0001, 0.00005]},
            "LAMBDA": {"values": [0.3, 0.6, 0.9]},
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
