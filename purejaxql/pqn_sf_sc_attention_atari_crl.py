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
from typing import Any, List, Tuple, Dict, Callable, Optional

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
from purejaxql.utils.consolidation_helpers import update_and_accumulate_tree, update_and_accumulate_task
from purejaxql.utils.similarity import rbf_similarity
from flax.core import freeze, unfreeze, FrozenDict

Params = FrozenDict


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
    sf_dim: int = 16

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
        basis_features = rep / jnp.linalg.norm(rep, ord=2, axis=-1, keepdims=True)

        task = jax.lax.stop_gradient(task)
        task_normalized = task / jnp.linalg.norm(task, ord=2, axis=-1, keepdims=True)
        rep_task = jnp.concatenate([rep, task_normalized], axis=-1)

        # features for SF
        features_critic_sf = nn.Dense(features=self.sf_dim)(rep_task)
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
    sf_dim: int
    num_actions: int
    num_beakers: int
    proj_factor: int = 1

    @nn.compact
    def __call__(self, basis_features_all, sf_all, task, mask):
        batch_size = sf_all.shape[0]

        sf_first = sf_all[:, :1, :, :]  # shape (batch, 1, ...)
        # sf_rest = jax.lax.stop_gradient(
        #     sf_all[:, 1:, :, :]
        # )  # shape (batch, num_beakers-1, ...)
        # sf_all = jnp.concatenate([sf_first, sf_rest], axis=1)

        sf_all_reshaped = jnp.reshape(
            sf_all, (batch_size, self.num_beakers, self.num_actions * self.sf_dim)
        )

        sf_first_reshaped = jnp.reshape(
            sf_first, (batch_size, 1, self.num_actions * self.sf_dim)
        )  # shape (batch, num_actions * sf_dim)

        print("input task shape: ", task.shape)

        # Normalize and tile task
        task = jax.lax.stop_gradient(task)
        task_normalized = task / jnp.linalg.norm(task, ord=2, axis=-1, keepdims=True)
        # task_normalized = jnp.tile(
        #     task_normalized, (1, self.num_beakers, 1)
        # )

        print("task_normalized shape: ", task_normalized.shape)
        print("sf_first shape: ", sf_first_reshaped.shape)

        print("sf_all_reshaped shape: ", sf_all_reshaped.shape)
        print("basis_features_all shape: ", basis_features_all.shape)


        """
        Compute similarity between the first beaker and the rest of the beakers using rbf similarity with task, basis 
        features and successor features as input.
        """
        # concat basis features and sf
        basis_features_sf_task = jnp.concatenate(
            [basis_features_all, sf_all_reshaped, task_normalized], axis=-1
        )
        print("basis_features_sf_task shape: ", basis_features_sf_task.shape)

        basis_features_sf_task_left = basis_features_sf_task[
            :, :-1, :
        ]  # shape (batch, num_beakers-1, (num_actions * sf_dim) + sf_dim + task_dim)
        basis_features_sf_task_right = basis_features_sf_task[
            :, 1:, :
        ]  # shape (batch, num_beakers-1, (num_actions * sf_dim) + sf_dim + task_dim)

        basis_features_sf_task_similarity = rbf_similarity(
            basis_features_sf_task_left, basis_features_sf_task_right
        ).mean(
            axis=0
        )  # shape (num_beakers-1)
        print(
            "basis_features_sf_task_similarity shape: ",
            basis_features_sf_task_similarity.shape,
        )

        basis_features_first = basis_features_all[:, :1, :]  # shape (batch, 1, ...)
        basis_features_rest = jax.lax.stop_gradient(basis_features_all[:, 1:, :])
        basis_features_all = jnp.concatenate(
            [basis_features_first, basis_features_rest], axis=1
        )

        # Attention mechanism. Using task, sf and basis features as query and keys. The values are the sf only.
        basis_features_sf_first_task = jnp.concatenate(
            [basis_features_first, sf_first_reshaped, task_normalized[:, :1, :]],
            axis=-1,
        )
        print("basis_features_sf_first_task shape:", basis_features_sf_first_task.shape)

        query = nn.Dense(
            features=self.sf_dim * 3 * self.proj_factor, name="query", use_bias=False
        )(basis_features_sf_first_task)
        print("query shape:", query.shape)

        basis_features_sf_all_task = jnp.concatenate(
            [basis_features_all, sf_all_reshaped, task_normalized], axis=-1
        )
        print("basis_features_sf_all_task shape:", basis_features_sf_all_task.shape)

        keys = nn.Dense(self.sf_dim * 3 * self.proj_factor)(
            basis_features_sf_all_task
        )  # (batch_size, num_beakers, d_model)
        values = nn.Dense(self.sf_dim * self.proj_factor)(
            sf_all
        )  # (batch_size, num_beakers, num_actions, d_model)

        # mask = jnp.reshape(mask, (batch_size, self.num_beakers * self.num_actions, -1))
        # mask = jnp.reshape(mask, (batch_size, self.num_beakers, -1))
        # mask = jnp.repeat(
        #     mask, self.proj_factor, axis=-1
        # )  # (batch_size, num_beakers * num_actions, sf_dim * 2)

        print("keys shape:", keys.shape)
        print("values shape:", values.shape)
        print("mask shape:", mask.shape)

        mask_repeat = jnp.repeat(mask, 3, axis=-1)

        print("mask_repeat shape:", mask_repeat.shape)

        keys_masked = keys * mask_repeat
        values_masked = values

        # print("mask: ", mask.shape)
        # print("basis_features_all shape: ", basis_features_all.shape)

        # query = query
        # keys_masked = basis_features_all * mask
        # values_masked = sf_all

        # print("query shape: ", query.shape)
        # print("keys mask shape: ", keys_masked.shape)
        # print("value mask shape: ", values_masked.shape)

        # Compute logits
        attn_logits = jnp.matmul(query, jnp.swapaxes(keys_masked, -2, -1)) / jnp.sqrt(
            self.sf_dim * self.proj_factor
        )
        # logits shape: (batch_size, num_actions, num_beakers * num_actions)

        attn_logits = jnp.where(attn_logits == 0, -1e9, attn_logits)

        # Compute attention weights
        attention_weights = nn.softmax(attn_logits, axis=-1)

        print("attention_weights shape:", attention_weights.shape)

        # Compute attention output
        # attended_sf = jnp.matmul(attention_weights, values_masked)

        attended_sf = jnp.einsum("bna,baqf->bnqf", attention_weights, values_masked)

        # print("attended_sf shape:", attended_sf.shape)
        # print("task shape: ", task.shape)

        # Compute Q-values
        task_first_network = task[:, 0, :]
        q_1 = jnp.einsum("bi,bnji->bj", task_first_network, attended_sf)

        return (
            q_1,
            attended_sf,
            attn_logits,
            attention_weights,
            keys_masked,
            values_masked,
            basis_features_sf_task_similarity,
        )


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
    consolidation_params_tree: Any = None
    capacity: Any = None
    g_flow: Any = None
    timescales: Any = None
    consolidation_networks: Any = None


class CustomTaskState(TrainState):
    consolidation_tasks: Any = None


@chex.dataclass
class MultiTrainState:
    network_state: CustomTrainState
    task_state: CustomTaskState
    attention_network_state: TrainState


def init_meta(rng, sf_dim, num_env) -> chex.Array:
    _, task_rng_key = jax.random.split(rng)
    task = jax.random.uniform(task_rng_key, shape=(sf_dim,))
    task = task / jnp.linalg.norm(task, ord=2)
    task = jnp.tile(task, (num_env, 1))
    return task


def create_agent(rng, config, max_num_actions, observation_space_shape):
    sf_network = SFNetwork(
        action_dim=max_num_actions,
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
        sf_dim=config["SF_DIM"],
    )

    init_x = jnp.zeros((1, *observation_space_shape))
    init_task = jnp.zeros((1, config["SF_DIM"]))
    print("init_task shape:", init_task.shape)
    network_variables = sf_network.init(rng, init_x, init_task, train=False)
    task_params = {
        "w": init_meta(rng, config["SF_DIM"], config["NUM_ENVS"] + config["TEST_ENVS"])
    }

    attention_network = SFAttentionNetwork(
        sf_dim=config["SF_DIM"],
        num_actions=max_num_actions,
        num_beakers=config["NUM_BEAKERS"],
    )

    init_basis_features_all = jnp.zeros((1, config["NUM_BEAKERS"], config["SF_DIM"]))

    init_sf_all = jnp.zeros(
        (1, config["NUM_BEAKERS"], max_num_actions, config["SF_DIM"])
    )
    init_mask = jnp.zeros((1, config["NUM_BEAKERS"], config["SF_DIM"]))

    init_task_all = jnp.zeros((1, config["NUM_BEAKERS"], config["SF_DIM"]))

    attention_network_variables = attention_network.init(
        rng, init_basis_features_all, init_sf_all, init_task_all, init_mask
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.radam(learning_rate=config["LR"]),
    )

    tx_task = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.radam(learning_rate=config["LR_TASK"]),
    )

    tx_attention = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.radam(learning_rate=config["LR"]),
    )

    # Initialize the consolidation parameters
    capacity = [1]
    g_flow = [2 ** (-config["FLOW_INIT_INDEX"] - 3)]
    g_delta_t = [g_flow[-1] * config["DELTA_T_CONSOLIDATION"]]
    storage_timescales = []  # timescales that are adapted using the ratio 2^k/g_1_2
    recall_timescales = []  # timescales that are adapted using the ratio 2^k/g_1_2
    consolidation_params_tree = {}
    consolidation_networks = []
    consolidation_tasks = {}

    for exp in range(1, config["NUM_BEAKERS"]):
        capacity.append(config["BEAKER_CAPACITY"] ** (exp + config["FLOW_INIT_INDEX"]))
        g_flow.append(2 ** (-config["FLOW_INIT_INDEX"] - exp - 3))
        g_delta_t.append(g_flow[-1] * config["DELTA_T_CONSOLIDATION"])

        # storage timescale should be faster than recall timescale (meaning smaller than recall timescale)
        # storage timescale  = capacity / g_1_2
        # recall timescale = capacity / g_k_k+1
        storage_timescales.append(
            int(capacity[exp] / g_flow[exp - 1] * config["DELTA_T_CONSOLIDATION"])
        )
        recall_timescales.append(int(capacity[exp] / g_flow[0]))

        if exp > 0:
            network = SFNetwork(
                action_dim=max_num_actions,
                norm_type=config["NORM_TYPE"],
                norm_input=config.get("NORM_INPUT", False),
                sf_dim=config["SF_DIM"],
            )

            init_x = jnp.zeros((1, *observation_space_shape))
            init_task = jnp.zeros((1, config["SF_DIM"]))
            network_variables = network.init(rng, init_x, init_task, train=False)
            consolidation_params_tree[f"network_{exp}"] = network_variables["params"]
            consolidation_networks.append(network)
            consolidation_tasks[f"network_{exp}"] = init_meta(
                rng, config["SF_DIM"], config["NUM_ENVS"] + config["TEST_ENVS"]
            )

    recall_timescales.append(
        int(
            config["BEAKER_CAPACITY"]
            ** (config["NUM_BEAKERS"] + config["FLOW_INIT_INDEX"])
            // g_flow[0]
        )
    )

    g_flow = jnp.array(g_flow)
    capacity = jnp.array(capacity)
    print(f"Capacity: {capacity}")
    print(f"storage g_flow: {g_flow[:-1]}")
    print(f"recall g_flow: {g_flow}")
    print(f"storage timescales: {storage_timescales}")
    print(f"recall timescales: {recall_timescales}")
    print(f"g_delta_t: {g_delta_t}")

    product = g_flow[0] * config["DELTA_T_CONSOLIDATION"]
    assert (
        product <= 0.1
    ), "g_1_2 * delta_t should be less than or equal to 0.1 to ensure stability of the system!!!"

    sf_network_state = CustomTrainState.create(
        apply_fn=sf_network.apply,
        params=network_variables["params"],
        batch_stats=network_variables["batch_stats"],
        tx=tx,
        capacity=capacity,
        g_flow=g_flow,
        timescales=recall_timescales,
        consolidation_params_tree=consolidation_params_tree,
    )

    task_state = CustomTaskState.create(
        apply_fn=sf_network.apply,
        params=task_params,
        tx=tx_task,
        consolidation_tasks=consolidation_tasks,
    )

    attention_network_state = TrainState.create(
        apply_fn=attention_network.apply,
        params=attention_network_variables["params"],
        tx=tx_attention,
    )

    return (
        MultiTrainState(
            network_state=sf_network_state,
            task_state=task_state,
            attention_network_state=attention_network_state,
        ),
        sf_network,
        attention_network,
    )


def make_train(config):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    all_updates = config["NUM_UPDATES"] * config["NUM_EXPOSURES"] * config["NUM_TASKS"]
    print(f"NUM_UPDATES: {int(all_updates)}")

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

    def train(rng, exposure, train_state, sf_network, attention_network, task_id):

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
        train_state.network_state = train_state.network_state.replace(
            exploration_updates=0
        )

        params_set_to_zero = unfreeze(
            jax.tree_util.tree_map(
                lambda x: jnp.zeros_like(x), unfreeze(train_state.network_state.params)
            )
        )

        def apply_single_beaker(params, obs, task, batch_stats):
            _, basis_features, sf = sf_network.apply(
                {"params": params, "batch_stats": batch_stats},
                obs,
                task,
                train=False,
                mutable=False,
            )
            return basis_features, sf  # shape: (batch, num_actions, sf_dim)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                (_, basis_features, sf) = sf_network.apply(
                    {
                        "params": train_state.network_state.params,
                        "batch_stats": train_state.network_state.batch_stats,
                    },
                    last_obs,
                    train_state.task_state.params["w"],
                    train=False,
                )

                params_beakers = [
                    train_state.network_state.consolidation_params_tree[f"network_{i}"]
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
                    train_state.task_state.params["w"],
                    (
                        num_beakers,
                        *train_state.task_state.params["w"].shape,
                    ),
                )  # [num_beakers, batch, task_dim]

                # Vectorized application of getting sf for each beaker
                basis_features_beakers, sf_beakers = jax.vmap(
                    apply_single_beaker, in_axes=(0, 0, 0, None)
                )(
                    params_beakers_stacked,
                    obs_tiled,
                    task_tiled,
                    train_state.network_state.batch_stats,
                )

                sf_all = jnp.concatenate([sf[None], sf_beakers], axis=0)
                basis_features_all = jnp.concatenate(
                    [basis_features[None], basis_features_beakers], axis=0
                )

                sf_all = jnp.transpose(
                    sf_all, (1, 0, 3, 2)
                )  # (batch_size, num_beakers, num_actions, sf_dim)

                basis_features_all = jnp.transpose(basis_features_all, (1, 0, 2))

                print("sf_all shape:", sf_all.shape)
                print("basis_features_all shape:", basis_features_all.shape)

                """
                Make a mask to mask out the beakers in the consolidation system which has timescales less than the current 
                gradstep. 
                """
                mask = (
                    jnp.asarray(train_state.network_state.timescales, dtype=np.uint32)
                    < train_state.network_state.grad_steps
                )
                mask = mask[:-1]  # remove the first beaker as the first beaker is the current task
                mask = jnp.insert(mask, 0, 1)
                mask = mask.astype(jnp.int32)
                # mask = mask.reshape(1, -1, 1, 1)
                mask = mask.reshape(1, -1, 1)

                print("mask shape before tiling:", mask.shape)

                # broadcast the mask to the shape of (batch_size, num_beakers-1, num_actions, sf_dim)
                # mask_tiled = jnp.broadcast_to(
                #     mask,
                #     (sf_all.shape[0], mask.shape[1], sf_all.shape[2], sf_all.shape[3]),
                # )

                mask_tiled = jnp.broadcast_to(
                    mask,
                    (
                        basis_features_all.shape[0],
                        mask.shape[1],
                        basis_features_all.shape[2],
                    ),
                )

                # grab all the w in train_state.task_state.params and train_state.task_state.consolidation_tasks
                tasks_all = [train_state.task_state.params["w"]]
                print("tasks_all shape:", train_state.task_state.params["w"].shape)
                for i in range(1, config["NUM_BEAKERS"]):
                    tasks_all.append(
                        train_state.task_state.consolidation_tasks[f"network_{i}"]
                    )
                    print("task shape temp:", tasks_all[-1].shape)
                tasks_all_arr = jnp.stack(tasks_all, axis=1)

                print("tasks_all_arr shape:", tasks_all_arr.shape)

                # attention network
                q_vals, _, _, _, _, _, _ = attention_network.apply(
                    {
                        "params": train_state.attention_network_state.params,
                    },
                    basis_features_all,
                    sf_all,
                    tasks_all_arr,
                    mask_tiled,
                )

                # different eps for each env
                _rngs = jax.random.split(rng_a, total_envs)
                eps = jnp.full(
                    config["NUM_ENVS"],
                    eps_scheduler(train_state.network_state.exploration_updates),
                )

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
                task_params_target = train_state.task_state.params["w"][
                    : -config["TEST_ENVS"], :
                ]

            train_state.network_state = train_state.network_state.replace(
                timesteps=train_state.network_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            train_state.network_state = train_state.network_state.replace(
                total_returns=train_state.network_state.total_returns
                + transitions.reward.sum()
            )  # update total returns count

            (_, last_basis_features, last_sf) = sf_network.apply(
                {
                    "params": train_state.network_state.params,
                    "batch_stats": train_state.network_state.batch_stats,
                },
                transitions.next_obs[-1],
                task_params_target,
                train=False,
            )

            params_beakers = [
                train_state.network_state.consolidation_params_tree[f"network_{i}"]
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

            last_basis_features_beakers, last_sf_beakers = jax.vmap(
                apply_single_beaker, in_axes=(0, 0, 0, None)
            )(
                params_beakers_stacked,
                obs_tiled,
                task_tiled,
                train_state.network_state.batch_stats,
            )

            last_sf_all = jnp.concatenate([last_sf[None], last_sf_beakers], axis=0)
            last_basis_features_all = jnp.concatenate(
                [last_basis_features[None], last_basis_features_beakers], axis=0
            )

            last_sf_all = jnp.transpose(
                last_sf_all, (1, 0, 3, 2)
            )  # (batch_size, num_beakers, num_actions, sf_dim)

            last_basis_features_all = jnp.transpose(last_basis_features_all, (1, 0, 2))
            print("last_sf_all shape:", last_sf_all.shape)

            """
            Make a mask to mask out the beakers in the consolidation system which has timescales less than the current time
            step. 
            """
            mask = (
                jnp.asarray(train_state.network_state.timescales, dtype=np.uint32)
                < train_state.network_state.grad_steps
            )
            mask = mask[:-1]  # remove the first beaker as the first beaker is the current task
            mask = jnp.insert(mask, 0, 1)
            mask = mask.astype(jnp.int32)
            # mask = mask.reshape(1, -1, 1, 1)
            mask = mask.reshape(1, -1, 1)
            # mask_tiled = jnp.broadcast_to(
            #     mask,
            #     (
            #         last_sf_all.shape[0],
            #         mask.shape[1],
            #         last_sf_all.shape[2],
            #         last_sf_all.shape[3],
            #     ),
            # )
            mask_tiled = jnp.broadcast_to(
                mask,
                (
                    last_basis_features_all.shape[0],
                    mask.shape[1],
                    last_basis_features_all.shape[2],
                ),
            )

            tasks_all_target = [task_params_target]
            print("task_params_target shape: ", task_params_target.shape)
            for i in range(1, config["NUM_BEAKERS"]):
                if config.get("TEST_DURING_TRAINING", False):
                    tasks_all_target.append(
                        train_state.task_state.consolidation_tasks[f"network_{i}"][
                            : -config["TEST_ENVS"], :
                        ]
                    )
                else:
                    tasks_all_target.append(
                        train_state.task_state.consolidation_tasks[f"network_{i}"]
                    )
                print("temp consolidation_tasks shape: ", train_state.task_state.consolidation_tasks[f"network_{i}"].shape)
            tasks_all_target_arr = jnp.stack(tasks_all_target, axis=1)

            print("tasks_all_target_arr shape:", tasks_all_target_arr.shape)

            # attention network
            last_q, _, _, _, _, _, _ = attention_network.apply(
                {
                    "params": train_state.attention_network_state.params,
                },
                last_basis_features_all,
                last_sf_all,
                tasks_all_target_arr,
                mask_tiled,
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

                    def _loss_fn(params, params_consolidation, mask):
                        (_, basis_features, sf), updates = sf_network.apply(
                            {
                                "params": params["sf"],
                                "batch_stats": train_state.network_state.batch_stats,
                            },
                            minibatch.obs,
                            train_state.task_state.params["w"][
                                : -config["TEST_ENVS"], :
                            ],
                            train=True,
                            mutable=["batch_stats"],
                        )  # (batch_size*2, num_actions)

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
                            train_state.task_state.params["w"][
                                : -config["TEST_ENVS"], :
                            ],
                            (
                                num_beakers,
                                *train_state.task_state.params["w"][
                                    : -config["TEST_ENVS"], :
                                ].shape,
                            ),
                        )  # [num_beakers, batch, task_dim]

                        # Vectorized application
                        basis_features_beakers, sf_beakers = jax.vmap(
                            apply_single_beaker, in_axes=(0, 0, 0, None)
                        )(
                            params_beakers_stacked,
                            obs_tiled,
                            task_tiled,
                            train_state.network_state.batch_stats,
                        )

                        sf_all = jnp.concatenate([sf[None], sf_beakers], axis=0)
                        basis_features_all = jnp.concatenate(
                            [basis_features[None], basis_features_beakers], axis=0
                        )

                        sf_all = jnp.transpose(
                            sf_all, (1, 0, 3, 2)
                        )  # (batch_size, num_beakers, num_actions, sf_dim)
                        basis_features_all = jnp.transpose(
                            basis_features_all, (1, 0, 2)
                        )

                        # mask = mask.reshape(1, -1, 1, 1)
                        mask = mask.reshape(1, -1, 1)
                        # mask_tiled = jnp.broadcast_to(
                        #     mask,
                        #     (
                        #         sf_all.shape[0],
                        #         mask.shape[1],
                        #         sf_all.shape[2],
                        #         sf_all.shape[3],
                        #     ),
                        # )
                        mask_tiled = jnp.broadcast_to(
                            mask,
                            (
                                basis_features_all.shape[0],
                                mask.shape[1],
                                basis_features_all.shape[2],
                            ),
                        )

                        tasks_all = [
                            train_state.task_state.params["w"][
                                : -config["TEST_ENVS"], :
                            ]
                        ]
                        for i in range(1, config["NUM_BEAKERS"]):
                            tasks_all.append(
                                train_state.task_state.consolidation_tasks[
                                    f"network_{i}"
                                ][: -config["TEST_ENVS"], :]
                            )
                        tasks_all_arr = jnp.stack(tasks_all, axis=1)

                        print("tasks_all_arr shape:", tasks_all_arr.shape)

                        # attention network
                        (
                            q_vals,
                            attended_sf,
                            attn_logits,
                            attention_weights,
                            keys,
                            values,
                            basis_features_sf_task_sim,
                        ) = attention_network.apply(
                            {
                                "params": params["attention"],
                            },
                            basis_features_all,
                            sf_all,
                            tasks_all_arr,
                            mask_tiled,
                        )

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, (
                            updates,
                            chosen_action_qvals,
                            basis_features,
                            attn_logits,
                            attention_weights,
                            keys,
                            values,
                            basis_features_sf_task_sim,
                            tasks_all,
                        )

                    def _reward_loss_fn(task_params, basis_features, reward):
                        task_params_train = task_params["w"][: -config["TEST_ENVS"], :]
                        predicted_reward = jnp.einsum(
                            "ij,ij->i", basis_features, task_params_train
                        )
                        loss = 0.5 * jnp.square(predicted_reward - reward).mean()

                        return loss

                    def _consolidation_update_fn(
                        params: List[Params],
                        params_set_to_zero: Params,
                        g_flow: chex.Array,
                        capacity: chex.Array,
                        num_beakers: int,
                        mask: chex.Array,
                        basis_features_sf_task_sim: Optional[chex.Array] = None,
                        tasks_all: Optional[chex.Array] = None,
                        task_params_set_to_zero: Optional[chex.Array] = None,
                    ) -> Tuple[List[Params], float]:
                        loss = 0.0
                        task_consolidation_loss = 0.0

                        # Updating first beaker from the second beaker
                        # scale_first = g_flow[0] / capacity[0]
                        # params[0], loss = update_and_accumulate_tree(
                        #     params[0], params[1], scale_first, loss
                        # )

                        # Middle beakers: 1 to num_beakers - 2
                        for i in range(1, num_beakers - 1):
                            scale_prev = g_flow[i - 1] / capacity[i]
                            basis_features_sf_task_sim_prev = jnp.maximum(
                                basis_features_sf_task_sim[i], scale_prev
                            )
                            scale_prev *= basis_features_sf_task_sim_prev
                            # scale_next = (g_flow[i] / capacity[i])

                            # Consolidate from previous beaker
                            params[i], loss = update_and_accumulate_tree(
                                params[i], params[i - 1], scale_prev, loss, config["DELTA_T_CONSOLIDATION"]
                            )

                            tasks_all[i], task_consolidation_loss = update_and_accumulate_task(
                                tasks_all[i], tasks_all[i - 1], scale_prev, task_consolidation_loss, config["DELTA_T_CONSOLIDATION"]
                            )

                            # # Recall from next beaker, conditionally
                            # def do_recall(p, l):
                            #     return update_and_accumulate_tree(
                            #         p, params[i + 1], scale_next, l
                            #     )
                            #
                            # def no_recall(p, l):
                            #     return p, l
                            #
                            # params[i], loss = jax.lax.cond(
                            #     mask[i] != 0,
                            #     do_recall,
                            #     no_recall,
                            #     params[i],
                            #     loss,
                            # )

                        # Last beaker
                        scale_last = g_flow[-1] / capacity[-1]
                        scale_second_last = g_flow[-2] / capacity[-1]
                        basis_features_sf_task_sim_second_last = jnp.maximum(
                            basis_features_sf_task_sim[-1], scale_second_last
                        )
                        scale_second_last *= basis_features_sf_task_sim_second_last

                        params[-1], loss = update_and_accumulate_tree(
                            params[-1], params_set_to_zero, scale_last, loss, config["DELTA_T_CONSOLIDATION"]
                        )

                        tasks_all[-1], task_consolidation_loss = update_and_accumulate_task(
                            tasks_all[-1], task_params_set_to_zero, scale_last, task_consolidation_loss, config["DELTA_T_CONSOLIDATION"]
                        )

                        # consolidate from second last beaker
                        params[-1], loss = update_and_accumulate_tree(
                            params[-1], params[-2], scale_second_last, loss, config["DELTA_T_CONSOLIDATION"]
                        )

                        tasks_all[-1], task_consolidation_loss = update_and_accumulate_task(
                            tasks_all[-1], tasks_all[-2], scale_second_last, task_consolidation_loss, config["DELTA_T_CONSOLIDATION"]
                        )

                        # compute the norm of the params using jax.tree_util.tree_map and sum the norms with jnp.sum and jnp.linalg.norm
                        params_norm = []
                        for p in params:
                            current_param_norm = optax.global_norm(p)

                            params_norm.append(current_param_norm)

                        task_norm = []
                        for i in range(config["NUM_BEAKERS"]):
                            current_task_norm = optax.global_norm(tasks_all[i])
                            task_norm.append(current_task_norm)

                        return params, loss, params_norm, tasks_all, task_consolidation_loss, task_norm

                    """
                    Make a mask to mask out the beakers in the consolidation system which has timescales less than the current time
                    step. 
                    """
                    mask = (
                        jnp.asarray(
                            train_state.network_state.timescales, dtype=np.uint32
                        )
                        < train_state.network_state.grad_steps
                    )
                    mask = mask[:-1]  # remove the first beaker as the first beaker is the current task
                    mask = jnp.insert(mask, 0, 1)
                    mask = mask.astype(jnp.int32)

                    # combined params so that the gradients are computed for both networks
                    combined_params = {
                        "sf": train_state.network_state.params,
                        "attention": train_state.attention_network_state.params,
                    }

                    (
                        loss,
                        (
                            updates,
                            qvals,
                            basis_features,
                            attn_logits,
                            attention_weights,
                            keys,
                            values,
                            basis_features_sf_task_sim,
                            tasks_all,
                        ),
                    ), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                        combined_params,
                        train_state.network_state.consolidation_params_tree,
                        mask,
                    )

                    train_state = train_state.replace(
                        network_state=train_state.network_state.apply_gradients(
                            grads=grads["sf"]
                        ),
                        attention_network_state=train_state.attention_network_state.apply_gradients(
                            grads=grads["attention"]
                        ),
                    )

                    train_state.network_state = train_state.network_state.replace(
                        grad_steps=train_state.network_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )

                    # update task params using reward prediction loss
                    old_task_params = train_state.task_state.params["w"]
                    basis_features = jax.lax.stop_gradient(basis_features)
                    reward_loss, grads_task = jax.value_and_grad(_reward_loss_fn)(
                        train_state.task_state.params, basis_features, minibatch.reward
                    )
                    train_state.task_state = train_state.task_state.apply_gradients(
                        grads=grads_task
                    )
                    new_task_params = train_state.task_state.params["w"]

                    task_params_diff = jnp.linalg.norm(
                        new_task_params - old_task_params, ord=2, axis=-1
                    )

                    # consolidation update
                    all_params = []
                    all_params.append(train_state.network_state.params)

                    # to account for the first beaker
                    basis_features_sf_task_sim = jnp.insert(
                        basis_features_sf_task_sim, 0, 1
                    )

                    # modify basis_features_sf_task_sim based on the mask, to allow consolidation to overwrite initialization
                    basis_features_sf_task_sim = jnp.where(
                        mask == 0,
                        jnp.ones_like(basis_features_sf_task_sim),
                        basis_features_sf_task_sim,
                    )

                    for i in range(1, config["NUM_BEAKERS"]):
                        all_params.append(
                            train_state.network_state.consolidation_params_tree[
                                f"network_{i}"
                            ]
                        )

                    tasks_all_consolidation = [train_state.task_state.params["w"]]
                    for i in range(1, config["NUM_BEAKERS"]):
                        tasks_all_consolidation.append(
                            train_state.task_state.consolidation_tasks[f"network_{i}"]
                        )

                    task_params_set_to_zero = jnp.zeros_like(train_state.task_state.params["w"])

                    (
                        network_params,
                        consolidation_loss,
                        params_norm,
                        tasks_all_consolidation,
                        task_consolidation_loss,
                        task_norm,
                    ) = _consolidation_update_fn(
                        params=all_params,
                        params_set_to_zero=params_set_to_zero,
                        g_flow=train_state.network_state.g_flow,
                        capacity=train_state.network_state.capacity,
                        num_beakers=config["NUM_BEAKERS"],
                        mask=mask,
                        basis_features_sf_task_sim=basis_features_sf_task_sim,
                        tasks_all=tasks_all_consolidation,
                        task_params_set_to_zero=task_params_set_to_zero,
                    )

                    # replace train_state params with the new params
                    train_state.network_state = train_state.network_state.replace(
                        params=network_params[0],
                        consolidation_params_tree={
                            f"network_{i}": network_params[i]
                            for i in range(1, config["NUM_BEAKERS"])
                        },
                    )

                    train_state.task_state = train_state.task_state.replace(
                        params={"w" : tasks_all_consolidation[0]},
                        consolidation_tasks={
                            f"network_{i}": tasks_all_consolidation[i]
                            for i in range(1, config["NUM_BEAKERS"])
                        }
                    )

                    return (train_state, rng), (
                        loss,
                        qvals,
                        reward_loss,
                        task_params_diff,
                        consolidation_loss,
                        params_norm,
                        attn_logits,
                        attention_weights,
                        keys,
                        values,
                        basis_features_sf_task_sim,
                        mask,
                        task_consolidation_loss,
                        task_norm,
                    )

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
                (train_state, rng), (
                    loss,
                    qvals,
                    reward_loss,
                    task_params_diff,
                    consolidation_loss,
                    params_norm,
                    attn_logits,
                    attention_weights,
                    keys,
                    values,
                    basis_features_sf_task_sim,
                    mask,
                    task_consolidation_loss,
                    task_norm,
                ) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )

                return (train_state, rng), (
                    loss,
                    qvals,
                    reward_loss,
                    task_params_diff,
                    consolidation_loss,
                    params_norm,
                    attn_logits,
                    attention_weights,
                    keys,
                    values,
                    basis_features_sf_task_sim,
                    mask,
                    task_consolidation_loss,
                    task_norm,
                )

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (
                loss,
                qvals,
                reward_loss,
                task_params_diff,
                consolidation_loss,
                params_norm,
                attn_logits,
                attention_weights,
                keys,
                values,
                basis_features_sf_task_sim,
                mask,
                task_consolidation_loss,
                task_norm,
            ) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state.network_state = train_state.network_state.replace(
                n_updates=train_state.network_state.n_updates + 1
            )
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
                "exposure": exposure,
                "task_id": task_id,
                "exploration_updates": train_state.network_state.exploration_updates,
                "total_returns": train_state.network_state.total_returns,
                "task_params_diff": task_params_diff.mean(),
                "extrinsic rewards": transitions.reward.mean(),
                "consolidation_loss": consolidation_loss.mean(),
                "task_consolidation_loss": task_consolidation_loss.mean(),
                "lr_task": config["LR_TASK"],
            }
            task_norm_arr = jnp.stack(task_norm, axis=-1)
            print("task norm arr shape: ", task_norm_arr.shape)
            print("mask shape: ", mask.shape)

            # add norm of each beaker params to metrics
            for idx, p in enumerate(params_norm):
                metrics[f"params_norm_{idx}"] = jnp.mean(p)

            # print("output attn logits shape: ", attn_logits.shape)
            # print("output attention weights shape: ", attention_weights.shape)
            # print("output keys shape: ", keys.shape)
            # print("output values shape: ", values.shape)

            # add 1 to the first index of basis_features_sf_task_sim to match the shape of the beakers
            # basis_features_sf_task_sim = jnp.insert(basis_features_sf_task_sim, 0, 1)

            for i in range(config["NUM_BEAKERS"]):
                print("keys shape: ", keys.shape)
                print("values shape: ", values.shape)

                metrics[f"attn_logits_{i}"] = attn_logits[..., i].mean()
                metrics[f"attention_weights_{i}"] = attention_weights[..., i].mean()
                metrics[f"basis_features_sf_task_sim_{i}"] = basis_features_sf_task_sim[
                    ..., i
                ].mean()
                metrics[f"mask_{i}"] = mask[..., i].mean()
                metrics[f"task_norm_{i}"] = task_norm_arr[..., i].mean()
                # metrics[f"keys_{i}"] = keys[..., i].mean()
                # metrics[f"values_{i}"] = values[..., i].mean()
                # metrics[f"task_norm_{i}"] = task_norm[i].mean()
                # metrics[f"mask_{i}"] = mask[i].mean()


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

        return {
            "runner_state": runner_state,
            "metrics": metrics,
            "train_state": runner_state[0],
        }

    return train


def single_run(config):

    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn")

    start_time = time.time()

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f'{config["ALG_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    # Get list of environments
    env_names = config["alg"]["ENV_NAME"]

    if config["alg"]["NUM_TASKS"] == 3:
        env_names = "Pong-v5, Breakout-v5, SpaceInvaders-v5"
    elif config["alg"]["NUM_TASKS"] == 5:
        env_names = "Alien-v5, Atlantis-v5, Boxing-v5, Breakout-v5, Centipede-v5"
        # env_names = "Alien-v5, Alien-v5, Alien-v5, Alien-v5, Alien-v5"

    elif config["alg"]["NUM_TASKS"] < 3:
        raise NotImplementedError("Less than 3 games not supported yet.")

    if isinstance(env_names, str):
        env_names = [e.strip() for e in env_names.split(",")]

    # Number of exposures to repeat the environments
    num_exposures = config["alg"].get("NUM_EXPOSURES", 1)

    # determine the max number of actions
    max_num_actions = 18  # atari has at most 18 actions
    observation_space_shape = (4, 84, 84)

    config["alg"]["TOTAL_TIMESTEPS_DECAY"] = (
        config["alg"]["TOTAL_TIMESTEPS_DECAY"] * config["NUM_TASKS"]
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rng, rng_agent = jax.random.split(rng)
    agent_train_state, sf_network, attention_network = create_agent(
        rng_agent, config, max_num_actions, observation_space_shape
    )

    for cycle in range(num_exposures):
        print(f"\n=== Cycle {cycle + 1}/{num_exposures} ===")
        for idx, env_name in enumerate(env_names):
            print(f"\n--- Running environment: {env_name} ---")
            task_id = cycle * config["alg"]["NUM_TASKS"] + idx
            config["ENV_NAME"] = env_name
            if config["NUM_SEEDS"] > 1:
                raise NotImplementedError("Vmapped seeds not supported yet.")
            else:
                # outs = jax.jit(make_train(config))(rng, exposure)
                outs = jax.jit(
                    lambda rng: make_train(config)(
                        rng,
                        cycle,
                        agent_train_state,
                        sf_network,
                        attention_network,
                        task_id,
                    )
                )(rng)
            print(f"Took {time.time()-start_time} seconds to complete.")

            agent_train_state = outs["train_state"]

            updates_per_task = (
                config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
            )

            total_updates_all_task = (
                updates_per_task * config["NUM_TASKS"] * num_exposures
            )

            assert (
                agent_train_state.network_state.timescales[0] <= updates_per_task
            ), "Storage timescale for the first task should not exceed the number of updates per task"

            assert (
                agent_train_state.network_state.timescales[-1] > total_updates_all_task
            ), "Storage timescale should be more than the total number of updates"

            # save params
            if config.get("SAVE_PATH", None) is not None:

                from purejaxql.utils.save_load import save_params

                model_state = outs["runner_state"][0]
                save_dir = os.path.join(config["SAVE_PATH"], env_name)
                os.makedirs(save_dir, exist_ok=True)
                OmegaConf.save(
                    config,
                    os.path.join(
                        save_dir,
                        f'{alg_name}_exposure{cycle}_task{idx}_seed{config["SEED"]}_config.yaml',
                    ),
                )

                # assumes not vmpapped seeds
                params = model_state.network_state.params
                save_path = os.path.join(
                    save_dir,
                    f'{alg_name}_exposure{cycle}_task{idx}_seed{config["SEED"]}.safetensors',
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
    # main()
    try:
        main()
    except Exception as e:
        import traceback

        print("Uncaught Exception in Hydra Job:")
        traceback.print_exc()
        raise e
