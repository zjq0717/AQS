# source (TD3 & TD3+BC): https://github.com/sfujim/TD3 & https://github.com/sfujim/TD3_BC
# https://arxiv.org/pdf/1802.09477 & https://arxiv.org/pdf/2106.06860
# source (SPOT): https://github.com/thuml/SPOT/tree/58c591dc48fbd9ff632b7494eab4caf778e86f4a
# https://arxiv.org/pdf/2202.06239.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import datetime
import torch.utils.tensorboard as thboard
import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import wandb
from collections import OrderedDict
from gym.spaces import Discrete
##
import time

TensorBatch = List[torch.Tensor]
ENVS_WITH_GOAL = ("antmaze")
log_dir = './logs/v1/'
class Hot_Plug(object):
	def __init__(self, model):
		self.model = model
		self.params = OrderedDict(self.model.named_parameters())
	def update(self, lr=0.1):
		for param_name in self.params.keys():
			path = param_name.split('.')
			cursor = self.model
			for module_name in path[:-1]:
				cursor = cursor._modules[module_name]
			if lr > 0:
				if self.params[param_name].requires_grad:
					cursor._parameters[path[-1]] = self.params[param_name] - lr*self.params[param_name].grad
			else:
				cursor._parameters[path[-1]] = self.params[param_name]
	def restore(self):
		self.update(lr=0)

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "hopper-medium-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 0  # Eval environment seed
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    offline_iterations: int = int(1e6)  # Number of offline updates
    online_iterations: int = int(1e6)  # Number of online updates
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    algo_type: str = "TD3"
    # TD3
    actor_lr: float = 1e-4  # Actor learning rate
    critic_lr: float = 3e-4  # Actor learning rate
    actor_ln: bool = False # Actor layer normalization
    critic_ln: bool = False # Critic layer normalization
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    normalize: bool = False  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # TD3 + BC
    alpha: float = 2.5 # Normalize BC term
    # SPOT VAE
    vae_lr: float = 1e-3  # VAE learning rate
    vae_hidden_dim: int = 750  # VAE hidden layers dimension
    vae_latent_dim: Optional[int] = None  # VAE latent space, 2 * action_dim if None
    beta: float = 0.5  # KL loss weight
    vae_iterations: int = 100_000  # Number of VAE training updates
    # SPOT
    actor_init_w: Optional[float] = None  # Actor head init parameter
    critic_init_w: Optional[float] = None  # Critic head init parameter
    lambd: float = 1.0  # Support constraint weight
    num_samples: int = 1  # Number of samples for density estimation
    iwae: bool = False  # Use IWAE loss
    lambd_cool: bool = False  # Cooling lambda during fine-tune
    lambd_end: float = 0.2  # Minimal value of lambda
    online_discount: float = 0.995  # Discount for online tuning
    # Wandb logging


    temperature: float = 5
    N_pretrain: float = 50000
    weight_lr: float = 3e-4
    N_tau: int = 25000
    kappa: float = 0.9
    kappa_cool: int = 150000
    kappa_end: float = 0.1
    UTD_ratio: int = 1

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
            
    def _set_custom_config(self, custom_config):
        for k, v in custom_config.items():
            if type(getattr(self, k)) is float:
                v = float(v)
            if type(getattr(self, k)) is int:
                v = int(v)
            if type(getattr(self, k)) is bool:
                v = bool(v)
            setattr(self, k, v)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._not_dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._not_dones[:n_transitions] = self._to_tensor(1 - data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices].to(self._device)
        actions = self._actions[indices].to(self._device)
        rewards = self._rewards[indices].to(self._device)
        next_states = self._next_states[indices].to(self._device)
        not_dones = self._not_dones[indices].to(self._device)
        return [states, actions, next_states, rewards, not_dones]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        not_done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._not_dones[self._pointer] = self._to_tensor(1. - not_done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)


class SegmentTree:
    def __init__(
        self, max_size, observation_dim, action_dim
    ):
        self._observations = np.zeros((max_size, observation_dim))
        self._next_obs = np.zeros((max_size, observation_dim))
        self._actions = np.zeros((max_size, action_dim))
        self._rewards = np.zeros((max_size, 1))
        self._terminals = np.zeros((max_size, 1), dtype="uint8")

        self.index = 0
        self.max_size = max_size
        self.full = False
        self.tree_start = 2 ** (max_size - 1).bit_length() - 1
        self.sum_tree = np.zeros((self.tree_start + self.max_size), dtype=np.float32)

        self.max = 1.0

    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        # [0,1,2,3] -> [1,3,5,7; 2,4,6,8]
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    def update(self, indices, values):
        self.sum_tree[indices] = values
        self._propagate(indices)
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # update single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # set new value
        self._propagate_index(index)  # propagate value
        self.max = max(value, self.max)

    def append(
        self,
        observation,
        action,
        next_obs,
        reward,
        terminal,
        value,
        **kwargs
    ):
        self._observations[self.index] = observation
        self._actions[self.index] = action
        self._next_obs[self.index] = next_obs
        self._rewards[self.index] = reward
        self._terminals[self.index] = terminal

        self._update_index(self.index + self.tree_start, value)
        self.index = (self.index + 1) % self.max_size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    def _retrieve(self, indices, values):
        children_indices = indices * 2 + np.expand_dims(
            [1, 2], axis=1
        )  # Make matrix of children indices
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(
            np.int32
        )  # Classify which values are in left or right branches
        successor_indices = children_indices[
            successor_choices, np.arange(indices.size)
        ]  # Use classification to index into the indices matrix
        successor_values = (
            values - successor_choices * left_children_values
        )  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)

    def get(self, data_index):
        batch = dict()
        batch["observations"] = self._observations[data_index]
        batch["next_observations"] = self._next_obs[data_index]
        batch["actions"] = self._actions[data_index]
        batch["rewards"] = self._rewards[data_index]
        batch["terminals"] = self._terminals[data_index]
        return batch

    def total(self):
        return self.sum_tree[0]


class PriorityReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transitions = SegmentTree(
			max_size, self.state_dim, self.action_dim
		)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def convert_D4RL(self, dataset):
        dataset_size = dataset["observations"].shape[0]
        for i in range(dataset_size):
            obs = dataset["observations"][i]
            new_obs = dataset["next_observations"][i]
            action = dataset["actions"][i]
            reward = dataset["rewards"][i]
            not_done = dataset["terminals"][i]
            self.add(obs, action, new_obs, reward, not_done)			

    def add(self, observation, action, next_observation, reward, terminal, **kwargs):
        if isinstance(self.action_dim, Discrete):
            new_action = np.zeros(self.action_dim)
            new_action[action] = 1
        else:
            new_action = action
        self.transitions.append(
                observation,
                action,
                next_observation,
                reward,
                1 - terminal,
                self.transitions.max,
            )

    def _get_transitions(self, idxs):
        transitions = self.transitions.get(data_index=idxs)
        return transitions

    def _get_samples_from_segments(self, batch_size, p_total):
        segment_length = p_total / batch_size
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        while not valid:
            samples = (
                np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts
            )
            probs, idxs, tree_idxs = self.transitions.find(samples)
            if np.all(probs != 0):
                valid = True
        batch = self._get_transitions(idxs)
        batch["idxs"] = idxs
        batch["tree_idxs"] = tree_idxs

        return batch

    def sample(self, batch_size):
        # return tree_idxs s.t. their values can be updated
        p_total = self.transitions.total()
        batch = self._get_samples_from_segments(batch_size, p_total)
        sample = (
			torch.FloatTensor(batch["observations"]).to(self.device),
			torch.FloatTensor(batch["actions"]).to(self.device),
			torch.FloatTensor(batch["next_observations"]).to(self.device),
			torch.FloatTensor(batch["rewards"]).to(self.device),
			torch.FloatTensor(batch["terminals"]).to(self.device),
			torch.FloatTensor(batch["idxs"]).to(self.device),
			torch.FloatTensor(batch["tree_idxs"]).to(self.device)
			)
        return sample

    def update_priorities(self, idxs, priorities):
        self.transitions.update(idxs, priorities)

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self.transitions.index

    def get_diagnostics(self):
        return OrderedDict([("size", self.transitions.index)])


def set_env_seed(env: Optional[gym.Env], seed: int):
    env.seed(seed)
    env.action_space.seed(seed)


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        set_env_seed(env, seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    episode_lengths = []
    successes = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        episode_length = 0
        goal_achieved = False
        while not done:
            action = actor.act(state, device)
            state, reward, done, env_infos = env.step(action)
            episode_reward += reward
            episode_length += 1
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    actor.train()
    return np.asarray(episode_rewards), np.mean(successes), np.mean(episode_lengths)


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def modify_reward_online(reward: float, env_name: str, **kwargs) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward -= 1.0
    return reward


def weights_init(m: nn.Module, init_w: float = 3e-3):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-init_w, init_w)
        m.bias.data.uniform_(-init_w, init_w)


class VAE(nn.Module):
    # Vanilla Variational Auto-Encoder

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        max_action: float,
        hidden_dim: int = 750,
    ):
        super(VAE, self).__init__()
        if latent_dim is None:
            latent_dim = 2 * action_dim
        self.encoder_shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std

    def importance_sampling_estimator(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        beta: float,
        num_samples: int = 500,
    ) -> torch.Tensor:
        # * num_samples correspond to num of samples L in the paper
        # * note that for exact value for \hat \log \pi_\beta in the paper
        # we also need **an expection over L samples**
        mean, std = self.encode(state, action)

        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_enc + std_enc * torch.randn_like(std_enc)  # [B x S x D]

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        mean_dec = self.decode(state, z)
        std_dec = np.sqrt(beta / 4)

        # Find q(z|x)
        log_qzx = td.Normal(loc=mean_enc, scale=std_enc).log_prob(z)
        # Find p(z)
        mu_prior = torch.zeros_like(z).to(self.device)
        std_prior = torch.ones_like(z).to(self.device)
        log_pz = td.Normal(loc=mu_prior, scale=std_prior).log_prob(z)
        # Find p(x|z)
        std_dec = torch.ones_like(mean_dec).to(self.device) * std_dec
        log_pxz = td.Normal(loc=mean_dec, scale=std_dec).log_prob(action)

        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
        ll = w.logsumexp(dim=-1) - np.log(num_samples)
        return ll

    def encode(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder_shared(torch.cat([state, action], -1))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std

    def decode(
        self,
        state: torch.Tensor,
        z: torch.Tensor = None,
    ) -> torch.Tensor:
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = (
                torch.randn((state.shape[0], self.latent_dim))
                .to(self.device)
                .clamp(-0.5, 0.5)
            )
        x = torch.cat([state, z], -1)
        return self.max_action * self.decoder(x)


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = F.relu(self.l3(q))
        return q


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        init_w: Optional[float] = None,
        layernorm: bool = False,
    ):
        super(Actor, self).__init__()

        head = nn.Linear(256, action_dim)
        if init_w is not None:
            weights_init(head, init_w)

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            head,
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class EnsembleCritic(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            ensemble_size: int,  # 集成的数量
            init_w: Optional[float] = None,
            layernorm: bool = False,
    ):
        super(EnsembleCritic, self).__init__()

        self.ensemble_size = ensemble_size

        # 使用多个头，每个头对应一个Critic网络
        self.head = nn.Linear(256, 1)
        if init_w is not None:
            weights_init(self.head, init_w)

        # 用 nn.ModuleList 来存储多个Critic网络
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.ReLU(),
                nn.LayerNorm(256) if layernorm else nn.Identity(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.LayerNorm(256) if layernorm else nn.Identity(),
                self.head,
            )
            for _ in range(ensemble_size)
        ])

        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        outputs = [net(sa) for net in self.nets]
        outputs = torch.cat(outputs, dim=0)
        outputs_ = outputs.view(10, 256, 1)
        return outputs_


class Critic(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        init_w: Optional[float] = None,
        layernorm: bool = False,
    ):
        super(Critic, self).__init__()

        head = nn.Linear(256, 1)
        if init_w is not None:
            weights_init(head, init_w)

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256) if layernorm else nn.Identity(),
            head,
        )

        self.staet_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        output = self.net(sa)
        return output

class TD3_AQS:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        AQS_critic_1: nn.Module,
        AQS_critic_1_optimizer: torch.optim.Optimizer,
        AQS_critic_2: nn.Module,
        AQS_critic_2_optimizer: torch.optim.Optimizer,
        weight: nn.Module,
        weight_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        alpha: float = 2.5,
        device: str = "cpu",
        temperature: int = 5,
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer
        self.AQS_critic_1 = AQS_critic_1
        self.AQS_critic_1_target = copy.deepcopy(AQS_critic_1)
        self.AQS_critic_1_optimizer = AQS_critic_1_optimizer
        self.AQS_critic_2 = AQS_critic_2
        self.AQS_critic_2_target = copy.deepcopy(AQS_critic_2)
        self.AQS_critic_2_optimizer = AQS_critic_2_optimizer
        self.hotplug_1 = Hot_Plug(self.AQS_critic_1)
        self.hotplug_2 = Hot_Plug(self.AQS_critic_2)
        
        self.weight = weight
        self.weight_optimizer = weight_optimizer

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        
        self.online_it = 0
        self.total_it = 0
        self.online_pretrain_it = 0
        self.temperature = temperature
        self.device = device
        
    def offline_train(self, replay_buffer, batch_size) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        reward = reward.unsqueeze(0).expand(10, -1, -1)
        not_done = not_done.unsqueeze(0).expand(10, -1, -1)
        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q


        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["Critic_Loss"] = critic_loss.item()
        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)

            q = self.critic_1(state, pi)
            q = torch.mean(q, dim=0)
            lmbda = self.alpha / q.abs().mean().detach()

            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)

            log_dict["Actor_Loss"] = actor_loss.item()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def online_train(self, replay_buffer, batch_size, kappa, offline_replay_buffer=None, online_replay_buffer=None) -> Dict[str, float]:
        log_dict = {}
        self.online_it += 1
        self.total_it += 1

        state, action, next_state, reward, not_done = online_replay_buffer.sample(batch_size)
        state_all, action_all, next_state_all, reward_all, not_done_all, idx, tree_idx = replay_buffer.sample(batch_size)
        reward_all = reward_all.unsqueeze(0).expand(10, -1, -1)
        not_done_all = not_done_all.unsqueeze(0).expand(10, -1, -1)


        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.AQS_critic_1_target(next_state, next_action)
            target_q2 = self.AQS_critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimates
        current_q1 = self.AQS_critic_1(state, action)
        current_q2 = self.AQS_critic_2(state, action)

        # Compute critic loss
        AQS_critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["AQS_critic_loss"] = AQS_critic_loss.item()
        # Optimize the critic
        self.AQS_critic_1_optimizer.zero_grad()
        self.AQS_critic_2_optimizer.zero_grad()
        AQS_critic_loss.backward()
        self.AQS_critic_1_optimizer.step()
        self.AQS_critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic_1(state, pi)
            mean_Q = torch.mean(Q, dim=0)
            std_Q = torch.std(Q, dim=0)
            optQ = self.AQS_critic_1(state, pi)
            abs_diff = torch.abs(optQ - mean_Q)
            diff = optQ - mean_Q
            on_Q = mean_Q + kappa*diff


            Q_new = torch.where(abs_diff > std_Q, on_Q, mean_Q)

            # q = (1 - kappa) * Q + kappa * optQ
            q = Q_new
            actor_loss = -q.mean()

            log_dict["Actor_Loss"] = actor_loss.item()
            log_dict["kappa"] = kappa
            log_dict["Q"] = Q.mean().item()
            log_dict["optQ"] = optQ.mean().item()
            log_dict["total_Q"] = q.mean().item()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.AQS_critic_1_target, self.AQS_critic_1, self.tau)
            soft_update(self.AQS_critic_2_target, self.AQS_critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    
    def online_pretrain(self, train_replay_buffer, test_replay_buffer, batch_size, N_pretrain):

       
        for _ in range(N_pretrain):
            self.online_pretrain_it += 1
            train_state, train_action, train_next_state, train_reward, train_not_done = test_replay_buffer.sample(batch_size)

            with torch.no_grad():
                # Select action according to actor and add clipped noise
                noise = (torch.randn_like(train_action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )

                train_next_action = (self.actor_target(train_next_state) + noise).clamp(
                    -self.max_action, self.max_action
                )

                # Compute the target Q value
                target_train_q1 = self.AQS_critic_1_target(train_next_state, train_next_action)
                target_train_q2 = self.AQS_critic_2_target(train_next_state, train_next_action)
                target_train_q = torch.min(target_train_q1, target_train_q2)
                target_train_q = train_reward + train_not_done * self.discount * target_train_q

            # Get current Q estimates
            current_train_q1 = self.AQS_critic_1(train_state, train_action)
            current_train_q2 = self.AQS_critic_2(train_state, train_action)

            # Compute critic loss
            critic_train_loss = F.mse_loss(current_train_q1, target_train_q) + F.mse_loss(current_train_q2, target_train_q)
            # Optimize the critic
            self.AQS_critic_1_optimizer.zero_grad()
            self.AQS_critic_2_optimizer.zero_grad()
            critic_train_loss.backward()
            self.AQS_critic_1_optimizer.step()
            self.AQS_critic_2_optimizer.step()

            if self.online_pretrain_it % self.policy_freq == 0:
                soft_update(self.AQS_critic_1_target, self.AQS_critic_1, self.tau)
                soft_update(self.AQS_critic_2_target, self.AQS_critic_2, self.tau)
    



    def save(self, env):

        filename = './model/' + "online" + f"/offline_model"
        torch.save(self.critic_1.state_dict(), filename + "_critic_1")
        torch.save(self.critic_1_optimizer.state_dict(), filename + "_critic_1_optimizer")
        torch.save(self.critic_1_target.state_dict(), filename + "_critic_1_target")

        torch.save(self.critic_2.state_dict(), filename + "_critic_2")
        torch.save(self.critic_2_optimizer.state_dict(), filename + "_critic_2_optimizer")
        torch.save(self.critic_2_target.state_dict(), filename + "_critic_2_target")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.actor_target.state_dict(), filename + "_actor_target")
        print("save")

    def load(self, env):
        filename = './model/' + env + f"/offline_model"
        self.critic_1.load_state_dict(torch.load(filename + "_critic_1"))
        self.critic_1_optimizer.load_state_dict(torch.load(filename + "_critic_1_optimizer"))
        self.critic_1_target.load_state_dict(torch.load(filename + "_critic_1_target"))

        self.critic_2.load_state_dict(torch.load(filename + "_critic_2"))
        self.critic_2_optimizer.load_state_dict(torch.load(filename + "_critic_2_optimizer"))
        self.critic_2_target.load_state_dict(torch.load(filename + "_critic_2_target"))

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target.load_state_dict(torch.load(filename + "_actor_target"))
        print("load")


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)
    eval_env = gym.make(config.env)

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)

    max_steps = env._max_episode_steps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # get dataset
    dataset = d4rl.qlearning_dataset(env)

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)

    online_replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    offline_replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, config.device)
    offline_replay_buffer.load_d4rl_dataset(dataset)
    priority_replay_buffer = PriorityReplayBuffer(state_dim, action_dim)
    priority_replay_buffer.convert_D4RL(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    set_env_seed(eval_env, config.eval_seed)
    
    if config.algo_type == "SPOT":
        vae = VAE(
            state_dim, action_dim, config.vae_latent_dim, max_action, config.vae_hidden_dim
        ).to(config.device)
        vae_optimizer = torch.optim.Adam(vae.parameters(), lr=config.vae_lr)

    weight = MLP(
        state_dim, action_dim
    ).to(config.device)
    weight_optimizer = torch.optim.Adam(weight.parameters(), lr=config.weight_lr)

    actor = Actor(state_dim, action_dim, max_action, config.actor_init_w, config.actor_ln).to(
        config.device
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    critic_1 = EnsembleCritic(state_dim, action_dim, 10, config.critic_init_w, config.critic_ln).to(config.device)
    # critic_1 = Critic(state_dim, action_dim, config.critic_init_w, config.critic_ln).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=config.critic_lr)
    critic_2 = EnsembleCritic(state_dim, action_dim, 10, config.critic_init_w, config.critic_ln).to(config.device)
    # critic_2 = Critic(state_dim, action_dim, config.critic_init_w, config.critic_ln).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=config.critic_lr)
    AQS_critic_1 = Critic(state_dim, action_dim, config.critic_init_w, config.critic_ln).to(config.device)
    AQS_critic_1_optimizer = torch.optim.Adam(AQS_critic_1.parameters(), lr=config.critic_lr)
    AQS_critic_2 = Critic(state_dim, action_dim, config.critic_init_w, config.critic_ln).to(config.device)
    AQS_critic_2_optimizer = torch.optim.Adam(AQS_critic_2.parameters(), lr=config.critic_lr)



    #-----------------------------------------------------------------------------------------
    if not os.path.exists(f"{log_dir}"):
        os.makedirs(f"{log_dir}")
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_dir = Path(f"{log_dir}_OQS_{config.env}_{config.seed}_online")
    summary_dir.mkdir(parents=True, exist_ok=True)

    filename_suffix = f"{config.env}_{current_time}"
    writer = thboard.SummaryWriter(str(summary_dir), filename_suffix=filename_suffix)
    # -----------------------------------------------------------------------------------------



    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "AQS_critic_1": AQS_critic_1,
        "AQS_critic_1_optimizer": AQS_critic_1_optimizer,
        "AQS_critic_2": AQS_critic_2,
        "AQS_critic_2_optimizer": AQS_critic_2_optimizer,
        "weight": weight,
        "weight_optimizer": weight_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,

        "temperature": config.temperature,
    }
    
    if config.algo_type == "TD3":
        kwargs.update({"alpha": config.alpha})
        trainer = TD3_AQS(**kwargs)
    elif config.algo_type == "SPOT":
        kwargs.update({"vae": vae,
                       "vae_optimizer": vae_optimizer,
                       "beta": config.beta,
                       "lambd": config.lambd,
                       "num_samples": config.num_samples,
                       "iwae": config.iwae,
                       "lambd_cool": config.lambd_cool,
                       "lambd_end": config.lambd_end,
                       "max_online_steps": config.online_iterations})
        trainer = AQS_SPOT(**kwargs)

    print(f"Training, Env: {config.env}, Seed: {seed}")


        
    # wandb_init(asdict(config))
    
    if config.algo_type == "SPOT":
        print("Training VAE")
        for t in range(int(config.vae_iterations)):
            batch = offline_replay_buffer.sample(config.batch_size)
            batch = [b.to(config.device) for b in batch]
            log_dict = trainer.vae_train(batch)
            log_dict["vae_iter"] = t
            # wandb.log(log_dict, step=trainer.total_it)
        vae.eval()

    state, done = env.reset(), False
    episode_return = 0
    episode_step = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []
    
    start = time.time()
    print("Offline pretraining")
    for t in range(int(config.offline_iterations) + int(config.online_iterations) + 1):
        if t == config.offline_iterations:
            print("Online tuning")
            if config.algo_type == "SPOT":
                trainer.discount = config.online_discount
                # Resetting optimizers
                trainer.actor_optimizer = torch.optim.Adam(
                    actor.parameters(), lr=config.actor_lr
                )
                trainer.critic_1_optimizer = torch.optim.Adam(
                    critic_1.parameters(), lr=config.critic_lr
                )
                trainer.critic_2_optimizer = torch.optim.Adam(
                    critic_2.parameters(), lr=config.critic_lr
                )
        online_log = {}
        log_dict = {}
        if t >= config.offline_iterations:
            episode_step += 1
            action = actor(
                torch.tensor(
                    state.reshape(1, -1), device=config.device, dtype=torch.float32
                )
            )
            noise = (torch.randn_like(action) * config.expl_noise).clamp(
                -config.noise_clip, config.noise_clip
            )
            action += noise
            action = torch.clamp(max_action * action, -max_action, max_action)
            action = action.cpu().data.numpy().flatten()
            next_state, reward, done, env_infos = env.step(action)

            if is_env_with_goal:
                if not goal_achieved:
                    goal_achieved = is_goal_reached(reward, env_infos)
            episode_return += reward
            real_done = False  # Episode can timeout which is different from done
            if done and episode_step < max_steps:
                real_done = True

            if config.normalize_reward:
                reward = modify_reward_online(reward, config.env, **reward_mod_dict)

            online_replay_buffer.add_transition(state, action, next_state, reward, real_done)
            priority_replay_buffer.add(state, action, next_state, reward, real_done)
            state = next_state
            
            if done:
                state, done = env.reset(), False
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if is_env_with_goal:
                    train_successes.append(goal_achieved)
                    online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                    online_log["train/is_success"] = float(goal_achieved)
                online_log["train/episode_return"] = episode_return
                normalized_return = eval_env.get_normalized_score(episode_return)
                online_log["train/d4rl_normalized_episode_return"] = (
                    normalized_return * 100.0
                )
                online_log["train/episode_length"] = episode_step
                episode_return = 0
                episode_step = 0
                goal_achieved = False
            
            if t >= (config.offline_iterations + config.N_tau):
                if t == (config.offline_iterations + config.N_tau):
                    train_timesteps = int(config.N_pretrain)
                    print("Online Pre-Training")
                    print("--------------------------")
                    model_log_dir = './model/' + config.env
                    if os.path.exists(model_log_dir):
                        print(f"The directory {model_log_dir} exists. Please specify a different one.")
                    else:
                        print(f"Creating directory {model_log_dir}")
                        log_dir_name = f"{model_log_dir}"
                        os.mkdir(log_dir_name)
                    # policy_file = Path(config.load_model)
                    print("--------------------------")
                    # trainer.save(config.env)
                    print("--------------------------")
                    trainer.load(config.env)
                    print("--------------------------")
                    trainer.online_pretrain(
                        offline_replay_buffer, online_replay_buffer, config.batch_size, train_timesteps
                        )
                for _ in range(int(1)):
                    if config.kappa_cool != 0:
                       
                        if t<(config.offline_iterations + config.N_tau + 200000):
                            kappa = (t-config.offline_iterations - config.N_tau )*0.000005
                        # kappa = 1
                        # else:
                        #     kappa = 1
                        # kappa = 0
                    else:
                        kappa = config.kappa
      
                    log_dict = trainer.online_train(
                        priority_replay_buffer, config.batch_size, kappa, offline_replay_buffer, online_replay_buffer
                    )
                    
     

        if t >= (config.offline_iterations + config.N_tau):
            log_dict.update(online_log)
        log_dict["offline_iter" if t < config.offline_iterations else "online_iter"] = (
            t if t < config.offline_iterations else t - config.offline_iterations
        )
       
        # Evaluate episode
        if t % config.eval_freq == 0:
            stop = time.time()
            print(f"Time steps: {t}")
            eval_scores, success_rate, episode_lengths = eval_actor(
                eval_env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            eval_log = {}
            eval_log["Step"] = (t if t < config.offline_iterations else t - config.offline_iterations)
            eval_log["Return"] = eval_score
            if hasattr(eval_env, "get_normalized_score"):
                normalized_score = eval_env.get_normalized_score(np.mean(eval_scores)) * 100.0
                eval_log["Normalized_Score"] = normalized_score
            eval_log["Episode_Length"] = episode_lengths.mean()
            eval_log["Training_Time"] = (stop - start)
            print("-----------------")
            trainer.save(config.env)
      
            if t >= config.offline_iterations and is_env_with_goal:
                eval_successes.append(success_rate)
     
            
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_score:.3f}"

            )
            writer.add_scalar("eval/normalized_returns", normalized_score, t)

            if config.checkpoints_path is not None:
                if t >= config.offline_iterations:
                    save_path = os.path.join(config.checkpoints_path, f"Step_{t}")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(trainer.state_dict(), os.path.join(save_path, "weight.pt"))
            start = time.time()
                
    # wandb.finish()

if __name__ == "__main__":
    train()