from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Normal

from labml import monit, tracker, logger
from labml.configs import FloatDynamicHyperParam, IntDynamicHyperParam
from labml_helpers.module import Module
from labml_nn.rl.ppo import ClippedPPOLoss, ClippedValueFunctionLoss

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def obs_to_torch(obs: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device)


class Actor(Module):

    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()

        self.activation = nn.ReLU()

        self.ln1 = nn.Linear(obs_dim, hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.Linear(hidden_dim, act_dim)
        self.ln4 = nn.Linear(hidden_dim, act_dim)

        self.device = "cpu"

    def forward(self, obs: np.ndarray):
        obs_torch = obs_to_torch(obs, self.device)
        h = self.activation(self.ln1(obs_torch))
        h = self.activation(self.ln2(h))
        act_mean = self.ln3(h)
        log_std = self.ln4(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        act_std = torch.exp(log_std)
        return act_mean, act_std

    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        if deterministic:
            return mean.detach().cpu().numpy(), 0.0
        dist = Normal(mean, std)
        action = dist.sample()
        return action.cpu().numpy(), dist.log_prob(action).sum(axis=-1).cpu().numpy()

    def log_prob(self, obs, actions):
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        actions = obs_to_torch(actions, self.device)
        return dist.log_prob(actions).sum(axis=-1), dist.entropy().mean(axis=-1)


class Critic(Module):

    def __init__(self, obs_dim, hidden_dim):
        super().__init__()

        self.activation = nn.ReLU()

        self.ln1 = nn.Linear(obs_dim, hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.Linear(hidden_dim, 1)

        self.device = "cpu"

    def forward(self, obs: np.ndarray):
        obs_torch = obs_to_torch(obs, self.device)
        h = self.activation(self.ln1(obs_torch))
        h = self.activation(self.ln2(h))
        value = self.ln3(h)
        return value


class Model(Module):
    """
    # Model
    """

    def __init__(self, automaton, hidden_dim, gpu_device):
        super().__init__()

        self.gpu_device = gpu_device
        self.action_dims = [mode.action_space.shape[0] for _, mode in automaton.modes.items()]
        self.obs_dims = [mode.observation_space.shape[0] for _, mode in automaton.modes.items()]

        self.actors = nn.ModuleDict({
            mname: Actor(mode.observation_space.shape[0], mode.action_space.shape[0], hidden_dim)
            for mname, mode in automaton.modes.items()
        })

        self.critics = nn.ModuleDict({
            mname: Critic(mode.observation_space.shape[0], hidden_dim)
            for mname, mode in automaton.modes.items()
        })

    def group_by_modes(self, obs: np.ndarray, modes: List[str]):
        obs_map = {}
        for mode, o in zip(modes, obs):
            if mode not in obs_map:
                obs_map[mode] = o.view(1, -1)
            else:
                obs_map[mode] = np.concatenate([obs_map[mode], o.view(1, -1)])
        return obs_map

    def ungroup(self, values: Dict[str, np.ndarray], modes: List[str]):
        ret_list = []
        mode_counts = {mode: 0 for mode in values}
        for mode in modes:
            ret_list.append(values[mode][mode_counts[mode]])
            mode_counts[mode] += 1
        return ret_list

    def get_actions(self, obs: np.ndarray, modes: List[str], deterministic=False):
        obs_map = self.group_by_modes(obs, modes)
        action_map = {}
        log_probs = {}
        for mode, o in obs_map:
            action_map[mode], log_probs[mode] = self.actors[mode].get_action(o, deterministic)
        return np.squeeze(np.array(self.ungroup(action_map, modes))), \
            np.squeeze(np.array(self.ungroup(log_probs, modes)))

    def get_values(self, obs: np.ndarray, modes: List[str]):
        obs_map = self.group_by_modes(obs, modes)
        value_map = {}
        for mode, o in obs_map:
            value_map[mode] = self.critics[mode](o)
        return torch.stack(self.ungroup(value_map, modes)).squeeze()

    def log_prob(self, obs: np.ndarray, modes: List[str], actions: np.ndarray):
        obs_map = self.group_by_modes(obs, modes)
        action_map = self.group_by_modes(actions, modes)
        log_probs = {}
        entropy = {}
        for mode in obs_map:
            log_probs[mode], entropy[mode] = self.actors[mode].log_prob(
                obs_map[mode], action_map[mode])
        return torch.stack(self.ungroup(log_probs, modes)).squeeze(), \
            torch.stack(self.ungroup(entropy, modes)).squeeze()

    def gpu(self):
        self.to(self.gpu_device)
        for mode in self.actors:
            self.actors[mode].device = self.gpu_device
            self.critics[mode].device = self.gpu_device

    def cpu(self):
        self.to("cpu")
        for mode in self.actors:
            self.actors[mode].device = "cpu"
            self.critics[mode].device = "cpu"

    def forward(self):
        raise NotImplementedError


class GAE:
    def __init__(self, gamma: float, lambda_: float):
        self.lambda_ = lambda_
        self.gamma = gamma

    def __call__(self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:

        # advantages table
        advantages = np.zeros((len(rewards),), dtype=np.float32)
        last_advantage = 0

        # $V(s_{t+1})$
        last_value = values[-1]
        mask = 1.0 - done

        for t in reversed(range(len(rewards))):
            # mask if episode completed after step $t$
            last_value = last_value * mask[t]
            last_advantage = last_advantage * mask[t]

            # $\delta_t$
            delta = rewards[t] + self.gamma * last_value - values[t]

            # $\hat{A_t} = \delta_t + \gamma \lambda \hat{A_{t+1}}$
            last_advantage = delta + self.gamma * self.lambda_ * last_advantage

            # update
            advantages[t] = last_advantage
            last_value = values[t]

        return advantages


class Trainer:
    """
    # Trainer
    """

    def __init__(self, *, model: Model, max_ep_len: int,
                 max_steps_in_mode: int, bonus: float,
                 updates: int, batch_size: int, batches: int,
                 epochs: IntDynamicHyperParam,
                 value_loss_coef: FloatDynamicHyperParam,
                 entropy_bonus_coef: FloatDynamicHyperParam,
                 clip_range: FloatDynamicHyperParam,
                 learning_rate: FloatDynamicHyperParam,
                 ):
        # #### Configurations

        self.epochs = epochs
        # number of updates
        self.updates = updates
        # number of mini batches
        self.batches = batches
        # total number of samples for a single update
        self.batch_size = batch_size
        # size of a mini batch
        self.mini_batch_size = self.batch_size // self.batches
        assert (self.batch_size % self.batches == 0)
        # max episode length
        self.max_ep_len = max_ep_len
        # maximum steps within mode
        self.max_steps_in_mode = max_steps_in_mode
        # bonus for subtask completion
        self.bonus = bonus

        # Value loss coefficient
        self.value_loss_coef = value_loss_coef
        # Entropy bonus coefficient
        self.entropy_bonus_coef = entropy_bonus_coef
        # Clipping range
        self.clip_range = clip_range
        # Learning rate
        self.learning_rate = learning_rate

        # #### Initialize

        # model
        self.model = model

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # GAE with $\gamma = 0.99$ and $\lambda = 0.95$
        self.gae = GAE(0.99, 0.95)

        # PPO Loss
        self.ppo_loss = ClippedPPOLoss()

        # Value Loss
        self.value_loss = ClippedValueFunctionLoss()

        # dims
        self.action_dim = self.model.action_dims[0]
        self.obs_dim = self.model.obs_dims[0]

    def sample(self, automaton, mode_list) -> Tuple[Dict[str, Any], float]:
        """
        # Sample data with current policy
        """

        rewards = np.zeros((self.batch_size,), dtype=np.float32)
        actions = np.zeros((self.batch_size, self.action_dim), dtype=np.float32)
        done = np.zeros((self.batch_size,), dtype=np.bool)
        obs = np.zeros((self.batch_size, self.obs_dim), dtype=np.float32)
        log_pis = np.zeros((self.batch_size,), dtype=np.float32)
        values = np.zeros((self.batch_size + 1,), dtype=np.float32)
        modes = ['' for _ in range(self.batch_size)]

        mnum = 0
        steps = 0
        cum_reward = 0.0
        discount = 1.0
        best_cum_reward = -1e9
        steps_in_mode = 0
        mode = automaton.modes[mode_list[mnum]]
        state = mode.end_to_end_reset()

        with torch.no_grad():

            for t in range(self.batch_size):

                obs[t] = mode.observe(state)
                actions[t], log_pis[t] = self.model.get_actions(obs[t].view(1, -1), [mode.name])
                values[t] = self.model.get_values(obs[t].view(
                    1, -1), [mode.name]).detach().cpu().numpy()
                modes[t] = mode.name

                next_state = mode.step(state, actions[t])
                rewards[t] = mode.reward(state, actions[t], next_state)

                steps += 1
                steps_in_mode += 1

                done[t] = steps >= self.max_ep_len or steps_in_mode >= self.max_steps_in_mode or \
                    not mode.is_safe(next_state)
                state = next_state

                if not done[t]:
                    for tr in automaton.transitions[mode.name]:
                        if tr.guard(state):
                            if (mnum + 1) < len(mode_list):
                                mnum += 1
                                mode = automaton.modes[mode_list[mnum]]
                                state = tr.jump(mode, state)
                                steps_in_mode = 0
                            else:
                                done[t] = True
                            rewards[t] += self.bonus
                            break

                cum_reward += discount * rewards[t]
                discount *= self.gae.gamma

                if done[t]:
                    mnum = 0
                    steps = 0
                    steps_in_mode = 0
                    mode = automaton.modes[mode_list[mnum]]
                    state = mode.end_to_end_reset()

                    best_cum_reward = max(best_cum_reward, cum_reward)
                    cum_reward = 0.0
                    discount = 1.0

        # Get value of after the final step
        if not done[-1]:
            values[self.batch_size] = self.model.get_values(obs[-1].view(
                1, -1), [mode.name]).detach().cpu().numpy()

        #
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values,
            'log_pis': log_pis,
            'done': done,
            'rewards': rewards,
            'modes': modes
        }

        return samples, best_cum_reward

    def train(self, samples: Dict[str, Any]):
        """
        # Train the model based on samples
        """
        samples["advantages"] = self.gae(samples["done"], samples["rewards"], samples["values"])

        for _ in range(self.epochs()):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            # for each mini batch
            for start in range(0, self.batch_size, self.mini_batch_size):
                # get mini batch
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                # train
                loss = self._calc_loss(mini_batch)

                # Set learning rate
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.learning_rate()
                # Zero out the previously calculated gradients
                self.optimizer.zero_grad()
                # Calculate gradients
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                # Update parameters based on gradients
                self.optimizer.step()

    @staticmethod
    def _normalize(adv: np.ndarray):
        """#### Normalize advantage function"""
        return (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

    def _calc_loss(self, samples: Dict[str, Any]) -> torch.Tensor:
        """
        # Calculate total loss
        """

        # $R_t$ returns sampled from $\pi_{\theta_{OLD}}$
        sampled_return = samples['values'] + samples['advantages']

        # $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$,
        # where $\hat{A_t}$ is advantages sampled from $\pi_{\theta_{OLD}}$.
        # Refer to sampling function in [Main class](#main) below
        #  for the calculation of $\hat{A}_t$.
        sampled_normalized_advantage = self._normalize(samples['advantages'])

        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$
        # and $V^{\pi_\theta}(s_t)$;
        #  we are treating observations as state
        value = self.model.get_values(samples['obs'], samples['modes'])

        # $-\log \pi_\theta (a_t|s_t)$, $a_t$ are actions sampled from $\pi_{\theta_{OLD}}$
        log_pi, entropy = self.model.log_prob(samples['obs'], samples['modes'], samples['actions'])

        # Calculate policy loss
        policy_loss = self.ppo_loss(
            log_pi, samples['log_pis'], sampled_normalized_advantage, self.clip_range())

        # Calculate Entropy Bonus
        #
        # $\mathcal{L}^{EB}(\theta) =
        #  \mathbb{E}\Bigl[ S\bigl[\pi_\theta\bigr] (s_t) \Bigr]$
        entropy_bonus = entropy.mean()

        # Calculate value function loss
        value_loss = self.value_loss(value, samples['values'], sampled_return, self.clip_range())

        # $\mathcal{L}^{CLIP+VF+EB} (\theta) =
        #  \mathcal{L}^{CLIP} (\theta) +
        #  c_1 \mathcal{L}^{VF} (\theta) - c_2 \mathcal{L}^{EB}(\theta)$
        loss = (policy_loss
                + self.value_loss_coef() * value_loss
                - self.entropy_bonus_coef() * entropy_bonus)

        # for monitoring
        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()
        print(loss, approx_kl_divergence)

        return loss

    def run_training_loop(self, automaton, mode_list):
        """
        # Run training loop
        """

        # last 100 episode information
        tracker.set_queue('reward', 100, True)
        tracker.set_queue('length', 100, True)

        for update in monit.loop(self.updates):
            # sample with current policy
            samples, _ = self.sample(automaton, mode_list)

            # train the model
            self.train(samples)

            # Save tracked indicators.
            tracker.save()
            # Add a new line to the screen periodically
            if (update + 1) % 1_000 == 0:
                logger.log()


if __name__ == "__main__":
    from hybrid_gym.envs import make_rooms_model

    automaton = make_rooms_model()

    # Configurations
    configs = {
        'max_ep_len': 150,
        'max_steps_in_mode': 25,
        'bonus': 50.,
        'updates': 10000,
        'epochs': IntDynamicHyperParam(8),
        'batch_size': 128 * 8,
        'batches': 4,
        'value_loss_coef': FloatDynamicHyperParam(0.5),
        'entropy_bonus_coef': FloatDynamicHyperParam(0.01),
        'clip_range': FloatDynamicHyperParam(0.1),
        'learning_rate': FloatDynamicHyperParam(1e-2, (0, 1e-2)),
    }

    # model
    model = Model(automaton, 64, 'cpu')

    # Initialize the trainer
    trainer = Trainer(model=model, **configs)

    # Run and monitor the experiment
    trainer.run_training_loop(automaton, ["left", "right", "straight"] * 2)
