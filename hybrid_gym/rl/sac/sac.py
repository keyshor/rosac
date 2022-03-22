from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import random
import os
import pickle
# import time
from hybrid_gym.rl.sac.core import MLPActorCritic, combined_shape
from hybrid_gym.model import Controller


def optimizer_to(optim, device):
    device = torch.device(device)
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.info_buf = [{} for _ in range(size)]
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = 'cpu'

    def store(self, obs, act, rew, next_obs, done, info):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.info_buf[self.ptr] = info
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32, reward_fn=None):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        if reward_fn is not None:
            info = [self.info_buf[j] for j in idxs]
            for i in range(batch_size):
                batch['rew'][i] = reward_fn.obs_reward(
                    batch['obs'][i], batch['act'][i], batch['obs2'][i], batch['rew'][i], info[i])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
                for k, v in batch.items()}


class MySAC:
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                        | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                        | of Q* for the provided observations
                                        | and actions. (Critical: make sure to
                                        | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                        | estimate of Q* for the provided observations
                                        | and actions. (Critical: make sure to
                                        | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                        | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                        | actions in ``a``. Importantly: gradients
                                        | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    def __init__(self, obs_space, act_space, hidden_dims=(256, 256),
                 steps_per_epoch=500, epochs=100, replay_size=int(1e6), gamma=0.99,
                 polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
                 update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
                 test_ep_len=1000, log_interval=40, min_alpha=0.2, alpha_decay=0.01,
                 gpu_device='cuda:0'):

        # compute dims
        obs_dim = obs_space.shape
        act_dim = act_space.shape[0]

        # initialize networks
        self.ac = MLPActorCritic(obs_space, act_space, hidden_dims)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # set hyperparams
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.log_interval = log_interval
        self.test_ep_len = test_ep_len
        self.min_alpha = min_alpha
        self.alpha_decay = alpha_decay
        self.gpu_device = gpu_device
        self.device = 'cpu'

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32, device=self.device),
                           deterministic)

    def test_agent(self, env_list, reward_fns, ep_num):
        print('\n')
        env_num = 0
        for env, reward_fn in zip(env_list, reward_fns):
            env_num += 1
            avg_reward = 0.
            for j in range(self.num_test_episodes):
                o, d, ep_ret, ep_len = env.reset(), False, 0, 0
                while not(d or (ep_len == self.test_ep_len)):
                    # Take deterministic actions at test time
                    a = self.get_action(o, True)
                    o2, r, d, info = env.step(a)
                    corrected_r = r
                    if reward_fn is not None:
                        corrected_r = reward_fn.obs_reward(o, a, o2, r, info)
                    ep_ret += corrected_r
                    ep_len += 1
                    o = o2
                avg_reward += ep_ret
            avg_reward /= self.num_test_episodes
            print('Average reward for env {} after {} episodes: {}'.format(
                env_num, ep_num, avg_reward))
        print('\n')

    def train(self, env_list, verbose=False, retrain=False, reward_fns=None):

        if reward_fns is None:
            reward_fns = [None for _ in range(len(env_list))]

        # Prepare for interaction with environment
        max_steps = self.steps_per_epoch * self.epochs
        steps = 0
        self.alpha = max(self.min_alpha, self.alpha-self.alpha_decay)

        for i in range(max_steps):
            env_num = random.choice(np.arange(len(env_list)))
            env = env_list[env_num]
            reward_fn = reward_fns[env_num]
            o, ep_ret, ep_len = env.reset(), 0, 0

            # Main loop: collect experience in env and update/log each epoch
            for _ in range(self.max_ep_len+1):

                # Until start_steps have elapsed, randomly sample actions
                # from a uniform distribution for better exploration. Afterwards,
                # use the learned policy.
                if steps > self.start_steps or retrain:
                    a = self.get_action(o)
                else:
                    a = env.action_space.sample()

                # Step the env
                o2, r, d, info = env.step(a)

                corrected_r = r
                if reward_fn is not None:
                    corrected_r = reward_fn.obs_reward(o, a, o2, r, info)
                ep_ret += corrected_r
                ep_len += 1
                steps += 1

                # Ignore the "done" signal if it comes from hitting the time
                # horizon (that is, when it's an artificial terminal signal
                # that isn't based on the agent's state)
                d = False if ep_len == self.max_ep_len else d

                # Store experience to replay buffer
                self.replay_buffer.store(o, a, r, o2, d, info)

                # Super critical, easy to overlook step: make sure to update
                # most recent observation!
                o = o2

                # Update handling
                if steps >= self.update_after and steps % self.update_every == 0:
                    for j in range(self.update_every):
                        batch = self.replay_buffer.sample_batch(
                            self.batch_size, reward_fn=reward_fn)
                        self.update(data=batch)

                # End of trajectory handling
                if d or (ep_len == self.max_ep_len):
                    break

            if verbose:
                print('Return at episode {}: {}'.format(i, ep_ret))

            if i % self.log_interval == 0 and verbose:
                self.test_agent(env_list, reward_fns, i)

            if steps > max_steps:
                break

        return steps

    def to(self, device):
        self.ac = self.ac.to(device=device)
        self.ac_targ = self.ac_targ.to(device=device)
        self.q_params = list(itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters()))
        self.replay_buffer.device = device
        self.device = device
        optimizer_to(self.pi_optimizer, device)
        optimizer_to(self.q_optimizer, device)

    def gpu(self):
        self.to(self.gpu_device)

    def cpu(self):
        self.to('cpu')

    def get_policy(self, deterministic=True, use_target=False):
        if use_target:
            return SACController(self.ac_targ, deterministic, self.device)
        else:
            return SACController(self.ac, deterministic, self.device)


class SACController(Controller):

    def __init__(self, ac: MLPActorCritic, deterministic=True, device='cpu'):
        self.ac = ac
        self.deterministic = deterministic

        # device is fixed to that of ac, needs to be manually updated if ac's device is changed.
        self.device = device

    def get_action(self, o: np.ndarray) -> np.ndarray:
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32, device=self.device),
                           self.deterministic)

    def save(self, name: str, path: str) -> None:
        # modifies device of ac, might affect other code depending on ac.
        if self.device != 'cpu':
            self.ac = self.ac.to('cpu')
            self.device = 'cpu'
        fh = open(os.path.join(path, name + '.pkl'), 'wb')
        pickle.dump(self, fh)

    def get_value_fn(self, copy_self=True):
        if not copy_self:
            return self
        if self.device != 'cpu':
            self.ac = self.ac.to('cpu')
        self_copy = deepcopy(self)
        if self.device != 'cpu':
            self_copy.ac = self_copy.ac.to(self.device)
            self.ac = self.ac.to(self.device)
        return self_copy

    def __call__(self, o: np.ndarray, deterministic=False) -> float:
        obs = torch.as_tensor(o, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            act, _ = self.ac.pi(obs, deterministic, False)
            val = torch.squeeze(torch.min(self.ac.q1(obs, act), self.ac.q2(obs, act)))
        return float(val.cpu().numpy())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Ant-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    torch.set_num_threads(torch.get_num_threads())

    env = gym.make(args.env)
    model = MySAC(env.observation_space, env.action_space,
                  hidden_dims=[args.hid]*args.l,
                  gamma=args.gamma, epochs=args.epochs,
                  max_ep_len=200)
    model.train([env], verbose=True)
