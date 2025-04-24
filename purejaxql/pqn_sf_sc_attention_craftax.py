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
from typing import Any, List, Tuple

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
from purejaxql.utils.consolidation_helpers import update_and_accumulate_tree
from flax.core import freeze, unfreeze, FrozenDict

Params = FrozenDict


class SFNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False
    feature_dim: int = 128
    sf_dim: int = 256
    hidden_size: int = 512
    num_layers: int = 2 # lesser than Q network since we use additional layers to construct SF

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
        basis_features = rep / jnp.linalg.norm(rep, ord=2, axis=-1, keepdims=True)

        task = jax.lax.stop_gradient(task)
        task_normalized = task / jnp.linalg.norm(task, ord=2, axis=-1, keepdims=True)
        rep_task = jnp.concatenate([rep, task_normalized], axis=-1)

        # features for SF
        features_critic_sf = nn.Dense(features=self.feature_dim)(rep_task)
        features_critic_sf = nn.relu(features_critic_sf)

        # SF
        sf = nn.Dense(features=self.sf_dim * self.action_dim)(features_critic_sf)
        sf_action = jnp.reshape(
            sf,
            (
                -1,
                self.sf_dim,
                self.action_dim,
            ),
        )  # (batch_size, sf_dim, action_dim)

        q_1 = jnp.einsum("bi, bij -> bj", task, sf_action).reshape(
            -1, self.action_dim
        )  # (batch_size, action_dim)

        return q_1, basis_features, sf_action

class SFAttentionNetwork(nn.Module):
    feature_dim: int
    sf_dim: int
    num_actions: int
    num_beakers: int

    @nn.compact
    def __call__(self, sf_all, task, mask):

        sf_first = sf_all[:, :1, :, :]  # shape (batch, 1, ...)
        sf_rest = jax.lax.stop_gradient(
            sf_all[:, 1:, :, :]
        )  # shape (batch, num_beakers-1, ...)
        sf_all = jnp.concatenate([sf_first, sf_rest], axis=1)
        sf_all_masked = sf_all * mask

        # Normalize and tile task
        task = jax.lax.stop_gradient(task)
        task_normalized = task / jnp.linalg.norm(task, ord=2, axis=-1, keepdims=True)
        task_normalized = jnp.tile(
            task_normalized[:, None, :], (1, self.num_beakers, 1)
        )

        # Attention mechanism
        query = nn.Dense(features=self.sf_dim, name="query", use_bias=False)(
            task_normalized[:, 0, :]
        )[:, None, :]

        # Different dense layers for each beaker to compute keys and values
        keys_per_beaker = []
        values_per_beaker = []
        for i in range(self.num_beakers):
            keys1_layer = nn.Dense(
                features=self.feature_dim, name=f"keys1_beaker_{i}"
            )

            keys2_layer = nn.Dense(
                features=self.sf_dim, name=f"keys2_beaker_{i}"
            )

            values1_layer = nn.Dense(
                features=self.feature_dim, name=f"values1_beaker_{i}"
            )

            values2_layer = nn.Dense(
                features=self.sf_dim, name=f"values2_beaker_{i}"
            )

            # Add mlp for keys and values so that the attention network can learn to transform the sf
            keys1 = keys1_layer(sf_all_masked[:, i, :, :])
            keys1 = nn.relu(keys1)

            values1 = values1_layer(sf_all_masked[:, i, :, :])
            values1 = nn.relu(values1)

            keys_per_beaker.append(
                keys2_layer(keys1)
            )  # Apply to each beaker's SF
            values_per_beaker.append(
                values2_layer(values1)
            )  # Apply to each beaker's SF

        # Stack the keys and values along the beaker dimension
        keys = jnp.stack(
            keys_per_beaker, axis=1
        )  # (batch_size, num_beakers, num_actions, sf_dim)
        values = jnp.stack(
            values_per_beaker, axis=1
        )  # (batch_size, num_beakers, num_actions, sf_dim)

        attn_logits = jnp.einsum("bqf,bnaf->bqna", query, keys) / jnp.sqrt(self.sf_dim)

        # replace zero logits with a large negative number so that they are ignored in the softmax
        attn_logits = jnp.where(attn_logits == 0, -1e9, attn_logits)

        attention_weights = jax.nn.softmax(attn_logits, axis=2)

        attended_sf = jnp.einsum("bqna,bnaf->bqaf", attention_weights, sf_all_masked)

        attended_sf = attended_sf.squeeze(1).swapaxes(1, 2)

        # Compute Q-values
        q_1 = jnp.einsum("bi,bij->bj", task, attended_sf)

        return q_1, attended_sf, attn_logits, attention_weights, keys, values


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
    consolidation_params_tree: Any = None
    capacity: Any = None
    g_flow: Any = None
    timescales: Any = None
    consolidation_networks: Any = None


@chex.dataclass
class MultiTrainState:
    network_state: CustomTrainState
    task_state: TrainState
    attention_network_state: TrainState

def make_train(config):

    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    print(f"NUM_UPDATES: {config['NUM_UPDATES']}")

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
            feature_dim=config["FEATURE_DIM"],
            hidden_size=config["HIDDEN_SIZE"],
            num_layers=config["NUM_LAYERS"],
        )

        attention_network = SFAttentionNetwork(
            feature_dim=config["FEATURE_DIM"],
            sf_dim=config["SF_DIM"],
            num_actions=env.action_space(env_params).n,
            num_beakers=config["NUM_BEAKERS"],
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

            init_sf_all = jnp.zeros(
                (1, config["NUM_BEAKERS"], env.action_space(env_params).n, config["SF_DIM"])
            )
            init_mask = jnp.zeros(
                (1, config["NUM_BEAKERS"], env.action_space(env_params).n, config["SF_DIM"])
            )
            attention_network_variables = attention_network.init(
                rng, init_sf_all, init_task, init_mask
            )

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            tx_task = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr_task),
            )

            tx_attention = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=config["LR"]),
            )

            # Initialize the consolidation parameters
            capacity = []
            g_flow = []
            timescales = []
            adapted_timescales = []  # timescales that are adapted using the ratio 2^k/g_1_2
            adapted_g_flow = []  # g_flow that are adapted using the ratio 2^(-k-3)

            for exp in range(config["NUM_BEAKERS"]):
                capacity.append(config["BEAKER_CAPACITY"] ** (exp + config["FLOW_INIT_INDEX"]))
                g_flow.append(2 ** (-config["FLOW_INIT_INDEX"] - exp - 3))
                timescales.append(int(capacity[exp] / g_flow[exp]))

                adapted_timescales.append(int(capacity[exp] / g_flow[0]))
                adapted_g_flow.append(2 ** (-1 - exp - 3))

            if config["CONSOLIDATE_EARLIER"]:
                timescales = adapted_timescales
                g_flow = adapted_g_flow

            print(f"timescales: {timescales[:-1]}")
            print(f"g_flow: {g_flow[:-1]}")
            print(f"Capacity: {capacity[:-1]}")

            g_flow = jnp.array(g_flow)
            capacity = jnp.array(capacity)

            consolidation_params_tree = {}
            consolidation_networks = []

            for i in range(1, config["NUM_BEAKERS"]):
                network_sc = SFNetwork(
                    action_dim=env.action_space(env_params).n,
                    norm_type=config["NORM_TYPE"],
                    norm_input=config.get("NORM_INPUT", False),
                    sf_dim=config["SF_DIM"],
                    feature_dim=config["FEATURE_DIM"],
                    hidden_size=config["HIDDEN_SIZE"],
                    num_layers=config["NUM_LAYERS"],
                )

                init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
                init_task = jnp.zeros((1, config["SF_DIM"]))
                network_variables = network_sc.init(rng, init_x, init_task, train=False)
                consolidation_params_tree[f"network_{i}"] = network_variables["params"]
                consolidation_networks.append(network_sc)

            network_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
                capacity=capacity,
                g_flow=g_flow,
                timescales=timescales,
                consolidation_params_tree=consolidation_params_tree,
            )

            task_state = TrainState.create(
                apply_fn=network.apply,
                params=task_params,
                tx=tx_task,
            )

            attention_network_state = TrainState.create(
                apply_fn=attention_network.apply,
                params=attention_network_variables["params"],
                tx=tx_attention,
            )

            return MultiTrainState(
                network_state=network_state,
                task_state=task_state,
                attention_network_state=attention_network_state,
            )

        rng, _rng = jax.random.split(rng)
        multi_train_state = create_agent(rng)

        params_set_to_zero = unfreeze(
            jax.tree_util.tree_map(
                lambda x: jnp.zeros_like(x), unfreeze(multi_train_state.network_state.params)
            )
        )

        def apply_single_beaker(params, obs, task, batch_stats):
            (_, _, sf) = network.apply(
                {"params": params, "batch_stats": batch_stats},
                obs,
                task,
                train=False,
                mutable=False,
            )
            return sf  # shape: (batch, num_actions, sf_dim)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            multi_train_state, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                # (q_vals, _, _) = network.apply(
                #     {
                #         "params": multi_train_state.network_state.params,
                #         "batch_stats": multi_train_state.network_state.batch_stats,
                #     },
                #     last_obs,
                #     task=multi_train_state.task_state.params["w"],
                #     train=False,
                # )

                (_, basis_features, sf) = network.apply(
                    {
                        "params": multi_train_state.network_state.params,
                        "batch_stats": multi_train_state.network_state.batch_stats,
                    },
                    last_obs,
                    multi_train_state.task_state.params["w"],
                    train=False,
                )

                params_beakers = [
                    multi_train_state.network_state.consolidation_params_tree[f"network_{i}"]
                    for i in range(1, config["NUM_BEAKERS"])
                ]

                # Convert list of dicts into a batched PyTree
                params_beakers_stacked = jax.tree_util.tree_map(
                    lambda *x: jnp.stack(x), *params_beakers
                )

                num_beakers = config["NUM_BEAKERS"] - 1  # because beaker 0 is excluded

                # Tile obs/task for each beaker
                obs_tiled = jnp.broadcast_to(
                    last_obs, (num_beakers, *last_obs.shape)
                )  # [num_beakers, batch, ...]
                task_tiled = jnp.broadcast_to(
                    multi_train_state.task_state.params["w"],
                    (
                        num_beakers,
                        *multi_train_state.task_state.params["w"].shape,
                    ),
                )  # [num_beakers, batch, task_dim]

                # Vectorized application of getting sf for each beaker
                sf_beakers = jax.vmap(apply_single_beaker, in_axes=(0, 0, 0, None))(
                    params_beakers_stacked, obs_tiled, task_tiled, multi_train_state.network_state.batch_stats
                )

                sf_all = jnp.concatenate([sf[None], sf_beakers], axis=0)
                sf_all = jnp.transpose(sf_all, (1, 0, 3, 2))  # (batch_size, num_beakers, num_actions, sf_dim)

                """
                Make a mask to mask out the beakers in the consolidation system which has timescales less than the current time
                step. 
                """
                mask = (
                        jnp.asarray(multi_train_state.network_state.timescales, dtype=np.uint32)
                        < multi_train_state.network_state.grad_steps
                )
                mask = mask[
                       :-1
                       ]  # remove the last column of the mask since the first beaker is always updated
                mask = jnp.insert(mask, 0, 1)
                mask = mask.astype(jnp.int32)
                mask = mask.reshape(1, -1, 1, 1)

                # broadcast the mask to the shape of (batch_size, num_beakers-1, num_actions, sf_dim)
                mask_tiled = jnp.broadcast_to(mask, (sf_all.shape[0], mask.shape[1], sf_all.shape[2], sf_all.shape[3]))

                # attention network
                q_vals, _, _, _, _, _ = attention_network.apply(
                    {
                        "params": multi_train_state.attention_network_state.params,
                    },
                    sf_all,
                    multi_train_state.task_state.params["w"],
                    mask_tiled,
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

            # (last_q, _, _) = network.apply(
            #     {
            #         "params": multi_train_state.network_state.params,
            #         "batch_stats": multi_train_state.network_state.batch_stats,
            #     },
            #     transitions.next_obs[-1],
            #     task=multi_train_state.task_state.params["w"],
            #     train=False,
            # )
            task_params_target = multi_train_state.task_state.params["w"]

            (_, _, last_sf) = network.apply(
                {
                    "params": multi_train_state.network_state.params,
                    "batch_stats": multi_train_state.network_state.batch_stats,
                },
                transitions.next_obs[-1],
                task_params_target,
                train=False,
            )

            params_beakers = [
                multi_train_state.network_state.consolidation_params_tree[f"network_{i}"]
                for i in range(1, config["NUM_BEAKERS"])
            ]

            # Convert list of dicts into a batched PyTree
            params_beakers_stacked = jax.tree_util.tree_map(
                lambda *x: jnp.stack(x), *params_beakers
            )

            num_beakers = config["NUM_BEAKERS"] - 1  # because beaker 0 is excluded

            # Tile obs/task for each beaker
            obs_tiled = jnp.broadcast_to(
                transitions.next_obs[-1], (num_beakers, *transitions.next_obs[-1].shape)
            )  # [num_beakers, batch, ...]
            task_tiled = jnp.broadcast_to(
                task_params_target, (num_beakers, *task_params_target.shape)
            )  # [num_beakers, batch, task_dim]

            last_sf_beakers = jax.vmap(apply_single_beaker, in_axes=(0, 0, 0, None))(
                params_beakers_stacked, obs_tiled, task_tiled, multi_train_state.network_state.batch_stats
            )

            last_sf_all = jnp.concatenate([last_sf[None], last_sf_beakers], axis=0)
            last_sf_all = jnp.transpose(last_sf_all, (1, 0, 3, 2))  # (batch_size, num_beakers, num_actions, sf_dim)

            """
            Make a mask to mask out the beakers in the consolidation system which has timescales less than the current time
            step. 
            """
            mask = (
                    jnp.asarray(multi_train_state.network_state.timescales, dtype=np.uint32)
                    < multi_train_state.network_state.grad_steps
            )
            mask = mask[
                   :-1
                   ]  # remove the last column of the mask since the first beaker is always updated
            mask = jnp.insert(mask, 0, 1)
            mask = mask.astype(jnp.int32)
            mask = mask.reshape(1, -1, 1, 1)
            mask_tiled = jnp.broadcast_to(mask, (
            last_sf_all.shape[0], mask.shape[1], last_sf_all.shape[2], last_sf_all.shape[3]))

            # attention network
            last_q, _, _, _, _, _ = attention_network.apply(
                {
                    "params": multi_train_state.attention_network_state.params,
                },
                last_sf_all,
                task_params_target,
                mask_tiled,
            )

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

                    def _loss_fn(params, params_consolidation, mask):

                        if config.get("Q_LAMBDA", False):
                            # (q_vals, basis_features, _), updates = network.apply(
                            #     {
                            #         "params": params,
                            #         "batch_stats": multi_train_state.network_state.batch_stats,
                            #     },
                            #     minibatch.obs,
                            #     train=True,
                            #     mutable=["batch_stats"],
                            #     task=multi_train_state.task_state.params["w"],
                            # )

                            (_, basis_features, sf), updates = network.apply(
                                {
                                    "params": params["sf"],
                                    "batch_stats": multi_train_state.network_state.batch_stats,
                                },
                                minibatch.obs,
                                train=True,
                                mutable=["batch_stats"],
                                task=multi_train_state.task_state.params["w"],
                            )

                            params_beakers = [
                                params_consolidation[f"network_{i}"]
                                for i in range(1, config["NUM_BEAKERS"])
                            ]

                            # Convert list of dicts into a batched PyTree
                            params_beakers_stacked = jax.tree_util.tree_map(
                                lambda *x: jnp.stack(x), *params_beakers
                            )

                            num_beakers = (
                                    config["NUM_BEAKERS"] - 1
                            )  # because beaker 0 is excluded

                            # Tile obs/task for each beaker
                            obs_tiled = jnp.broadcast_to(
                                minibatch.obs, (num_beakers, *minibatch.obs.shape)
                            )  # [num_beakers, batch, ...]
                            task_tiled = jnp.broadcast_to(
                                multi_train_state.task_state.params["w"],
                                (
                                    num_beakers,
                                    *multi_train_state.task_state.params["w"].shape,
                                ),
                            )  # [num_beakers, batch, task_dim]

                            # Vectorized application
                            sf_beakers = jax.vmap(apply_single_beaker, in_axes=(0, 0, 0, None))(
                                params_beakers_stacked, obs_tiled, task_tiled, train_state.network_state.batch_stats
                            )

                            sf_all = jnp.concatenate([sf[None], sf_beakers], axis=0)
                            sf_all = jnp.transpose(sf_all,
                                                   (1, 0, 3, 2))  # (batch_size, num_beakers, num_actions, sf_dim)

                            mask = mask.reshape(1, -1, 1, 1)
                            mask_tiled = jnp.broadcast_to(mask, (
                                sf_all.shape[0], mask.shape[1], sf_all.shape[2], sf_all.shape[3]))

                            # attention network
                            (
                                q_vals,
                                attended_sf,
                                attn_logits,
                                attention_weights,
                                keys,
                                values,
                            ) = attention_network.apply(
                                {
                                    "params": params["attention"],
                                },
                                sf_all,
                                multi_train_state.task_state.params["w"],
                                mask_tiled,
                            )


                        else:
                            # if not using q_lambda, re-pass the next_obs through the network to compute target
                            (all_q_vals, basis_features, _), updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": multi_train_state.network_state.batch_stats,
                                },
                                jnp.concatenate((minibatch.obs, minibatch.next_obs)),
                                train=True,
                                mutable=["batch_stats"],
                                task=jnp.concatenate((multi_train_state.task_state.params["w"], multi_train_state.task_state.params["w"])),
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

                        return loss, (updates, chosen_action_qvals, basis_features)

                    def _reward_loss_fn(task_params, basis_features):
                        if config.get("Q_LAMBDA", False):
                            reward = minibatch.reward
                        else:
                            reward = jnp.concatenate((minibatch.reward, minibatch.reward))
                            task_params = jnp.concatenate((task_params["w"], task_params["w"]))
                        predicted_reward = jnp.einsum("ij,ij->i", basis_features, task_params)
                        loss = 0.5 * jnp.square(predicted_reward - reward).mean()

                        return loss

                    def _consolidation_update_fn(
                        params: List[Params],
                        params_set_to_zero: Params,
                        g_flow: chex.Array,
                        capacity: chex.Array,
                        mask: chex.Array,
                        num_beakers: int,
                    ) -> Tuple[List[Params], float]:
                        loss = 0.0

                        # First beaker
                        scale_first = g_flow[0] / capacity[0]
                        params[0], loss = update_and_accumulate_tree(
                            params[0], params[1], scale_first, loss
                        )

                        # Last beaker
                        scale_last = g_flow[-1] / capacity[-1]
                        scale_second_last = g_flow[-2] / capacity[-1]

                        params[-1], loss = update_and_accumulate_tree(
                            params[-1], params_set_to_zero, scale_last, loss
                        )
                        params[-1], loss = update_and_accumulate_tree(
                            params[-1], params[-2], scale_second_last, loss
                        )

                        # Middle beakers: 1 to num_beakers - 2
                        for i in range(1, num_beakers - 1):
                            scale_prev = g_flow[i - 1] / capacity[i]
                            scale_next = g_flow[i] / capacity[i]

                            # Consolidate from previous beaker
                            params[i], loss = update_and_accumulate_tree(
                                params[i], params[i - 1], scale_prev, loss
                            )

                            # Recall from next beaker, conditionally
                            def do_recall(p, l):
                                return update_and_accumulate_tree(
                                    p, params[i + 1], scale_next, l
                                )

                            def no_recall(p, l):
                                return p, l

                            params[i], loss = jax.lax.cond(
                                mask[i] != 0,
                                do_recall,
                                no_recall,
                                params[i],
                                loss,
                            )

                        # compute the norm of the params using jax.tree_util.tree_map and sum the norms with jnp.sum and jnp.linalg.norm
                        params_norm = []
                        for p in params:
                            current_param_norm = optax.global_norm(p)

                            params_norm.append(current_param_norm)


                        return params, loss, params_norm

                    """
                    Make a mask to mask out the beakers in the consolidation system which has timescales less than the current time
                    step. 
                    """
                    mask = (
                            jnp.asarray(multi_train_state.network_state.timescales, dtype=np.uint32)
                            < multi_train_state.network_state.timesteps
                    )
                    mask = mask[
                           :-1
                           ]  # remove the last column of the mask since the first beaker is always updated
                    mask = jnp.insert(mask, 0, 1)
                    mask = mask.astype(jnp.int32)

                    # combined params so that the gradients are computed for both networks
                    combined_params = {
                        "sf": multi_train_state.network_state.params,
                        "attention": multi_train_state.attention_network_state.params,
                    }

                    (loss, (updates, qvals, basis_features)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(combined_params,
                        multi_train_state.network_state.consolidation_params_tree,
                        mask,)
                    multi_train_state.network_state = multi_train_state.network_state.apply_gradients(grads=grads)
                    multi_train_state.network_state = multi_train_state.network_state.replace(
                        grad_steps=multi_train_state.network_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )

                    # update task params using reward prediction loss
                    old_task_params = multi_train_state.task_state.params["w"]
                    basis_features = jax.lax.stop_gradient(basis_features)
                    reward_loss, grads_task = jax.value_and_grad(
                        _reward_loss_fn
                    )(multi_train_state.task_state.params, basis_features)
                    multi_train_state.task_state = multi_train_state.task_state.apply_gradients(grads=grads_task)
                    new_task_params = multi_train_state.task_state.params["w"]

                    task_params_diff = jnp.linalg.norm(new_task_params - old_task_params, ord=2, axis=-1)

                    # consolidation update
                    all_params = []
                    all_params.append(multi_train_state.network_state.params)

                    for i in range(1, config["NUM_BEAKERS"]):
                        all_params.append(multi_train_state.network_state.consolidation_params_tree[f"network_{i}"])

                    network_params, consolidation_loss, params_norm = _consolidation_update_fn(
                        params=all_params,
                        params_set_to_zero=params_set_to_zero,
                        g_flow=multi_train_state.network_state.g_flow,
                        capacity=multi_train_state.network_state.capacity,
                        mask=mask,
                        num_beakers=config["NUM_BEAKERS"],
                    )

                    # replace train_state params with the new params
                    multi_train_state.network_state = multi_train_state.network_state.replace(
                        params=network_params[0],
                        consolidation_params_tree={
                            f"network_{i}": network_params[i]
                            for i in range(1, config["NUM_BEAKERS"])
                        },
                    )

                    return (multi_train_state, rng), (loss, qvals, reward_loss, task_params_diff, consolidation_loss, params_norm)

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
                (multi_train_state, rng), (loss, qvals, reward_loss, task_params_diff, consolidation_loss, params_norm) = jax.lax.scan(
                    _learn_phase, (multi_train_state, rng), (minibatches, targets)
                )

                return (multi_train_state, rng), (loss, qvals, reward_loss, task_params_diff, consolidation_loss, params_norm)

            rng, _rng = jax.random.split(rng)
            (multi_train_state, rng), (loss, qvals, reward_loss, task_params_diff, consolidation_loss, params_norm) = jax.lax.scan(
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
                "lr": lr_scheduler(multi_train_state.network_state.n_updates),
                "reward_loss": reward_loss.mean(),
                "task_params_diff": task_params_diff.mean(),
                "extrinsic rewards": transitions.reward.mean(),
                "consolidation_loss": consolidation_loss.mean(),
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
                        wandb.log(metrics, step=metrics["update_steps"])

                        for k, v in metrics.items():
                            print(f"{k}: {v}")

                jax.debug.callback(callback, metrics, original_rng)

            runner_state = (multi_train_state, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        def get_test_metrics(multi_train_state, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)
                q_vals, _, _ = network.apply(
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
