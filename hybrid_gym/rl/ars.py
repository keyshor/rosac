from hybrid_gym.rl.util import discounted_reward, get_rollout, test_policy
from hybrid_gym.model import Controller

import numpy as np
import pickle
import torch
import os


class ARSModel:

    def __init__(self, nn_params, ars_params, use_gpu=False):
        self.nn_policy = NNPolicy(nn_params, use_gpu)
        self.ars_params = ars_params
        self.mu_sum = None
        self.sigma_sq_sum = None
        self.n_states = 0

    def learn(self, env_list, verbose=False, reward_fns=None):
        best_policy, log_info, self.mu_sum, self.sigma_sq_sum, self.n_states = ars(
            env_list, self.nn_policy, self.ars_params, verbose=verbose, reward_fns=reward_fns,
            mu_sum=self.mu_sum, sigma_sq_sum=self.sigma_sq_sum, n_states=self.n_states)
        return best_policy, log_info

    def cpu(self):
        self.nn_policy = self.nn_policy.set_use_cpu()

    def gpu(self):
        self.nn_policy = self.nn_policy.set_use_gpu()

    def get_policy(self):
        return self.nn_policy


# Parameters for training a policy neural net.
#
# state_dim: int (n)
# action_dim: int (p)
# hidden_dim: int
# dir: str
# fname: str
class NNParams:
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.hidden_dim = hidden_dim


# Parameters for augmented random search policy.
#
# n_iters: int (ending condition)
# n_samples: int (N)
# n_top_samples: int (b)
# delta_std (nu)
# lr: float (alpha)
class ARSParams:
    def __init__(self, n_iters, n_samples, n_top_samples, delta_std, lr,
                 gamma, max_timesteps, track_best=False):
        self.n_iters = n_iters
        self.n_samples = n_samples
        self.n_top_samples = n_top_samples
        self.delta_std = delta_std
        self.lr = lr
        self.gamma = gamma
        self.timesteps = max_timesteps
        self.track_best = track_best


# Neural network policy.
class NNPolicy(Controller):
    # Initialize the neural network.
    #
    # params: NNParams
    def __init__(self, params, use_gpu=False):
        # Step 1: Parameters
        self.params = params
        self.use_gpu = use_gpu

        # Step 2: Construct neural network

        # Step 2a: Construct the input layer
        self.input_layer = torch.nn.Linear(
            self.params.state_dim, self.params.hidden_dim)

        # Step 2b: Construct the hidden layer
        self.hidden_layer = torch.nn.Linear(
            self.params.hidden_dim, self.params.hidden_dim)

        # Step 2c: Construct the output layer
        self.output_layer = torch.nn.Linear(
            self.params.hidden_dim, self.params.action_dim)

        # Step 2d: GPU settings
        if self.use_gpu:
            self.input_layer = self.input_layer.cuda()
            self.hidden_layer = self.hidden_layer.cuda()
            self.output_layer = self.output_layer.cuda()

        # Step 3: Construct input normalization
        self.mu = np.zeros(self.params.state_dim)
        self.sigma_inv = np.ones(self.params.state_dim)

    # Get the action to take in the current state.
    #
    # state: np.array([state_dim])
    def get_action(self, state):
        # Step 1: Normalize state
        state = (state - self.mu) * self.sigma_inv

        # Step 2: Convert to torch
        state = torch.tensor(state, dtype=torch.float)
        if self.use_gpu:
            state = state.cuda()

        # Step 3: Apply the input layer
        hidden = torch.relu(self.input_layer(state))

        # Step 4: Apply the hidden layer
        hidden = torch.relu(self.hidden_layer(hidden))

        # Step 5: Apply the output layer
        output = torch.tanh(self.output_layer(hidden))

        # Step 7: Convert to numpy
        actions = output.cpu().detach().numpy()

        # Step 6: Scale the outputs
        actions = self.params.action_bound * actions

        return actions

    # Construct the set of parameters for the policy.
    #
    # nn_policy: NNPolicy
    # return: list of torch parameters
    def parameters(self):
        parameters = []
        parameters.extend(self.input_layer.parameters())
        parameters.extend(self.hidden_layer.parameters())
        parameters.extend(self.output_layer.parameters())
        return parameters

    def set_use_cpu(self):
        self.use_gpu = False
        self.input_layer = self.input_layer.cpu()
        self.hidden_layer = self.hidden_layer.cpu()
        self.output_layer = self.output_layer.cpu()
        return self

    def set_use_gpu(self):
        self.use_gpu = True
        self.input_layer = self.input_layer.cuda()
        self.hidden_layer = self.hidden_layer.cuda()
        self.output_layer = self.output_layer.cuda()
        return self

    def save(self, name: str, path: str):
        use_gpu = False
        obj = self
        if self.use_gpu:
            use_gpu = True
            obj = self.set_use_cpu()
        fh = open(os.path.join(path, name + '.pkl'), 'wb')
        pickle.dump(obj, fh)
        if use_gpu:
            self.set_use_gpu()

    @classmethod
    def load(cls, name: str, path: str, use_gpu: bool = False, **kwargs):
        fh = open(os.path.join(path, name + '.pkl'), 'rb')
        policy = pickle.load(fh)
        if use_gpu:
            policy = policy.set_use_gpu()
        return policy


# Run augmented random search.
#
# env: Environment
# nn_policy: NNPolicy
# params: ARSParams
# gamma: discount factor for computing cumulative rewards
# use_envs_cum_reward: uses cum_reward() function for the environment when set to True
# sparse_rewards: adds a satisfaction probability value to log when set to true
#                 (works only for environments with sparse rewards and
#                  use_envs_cum_reward has to be False)
# process_id: string to distiguish console ouputs for simultaneous executions
def ars(env_list, nn_policy, params, verbose=False, reward_fns=None,
        mu_sum=None, sigma_sq_sum=None, n_states=0):
    # Step 1: Save original policy
    nn_policy_orig = nn_policy
    log_info = []
    best_reward = -1e9
    best_policy = nn_policy
    num_transitions = 0

    if reward_fns is None:
        reward_fns = [None for _ in range(len(env_list))]

    # Step 2: Initialize state distribution estimates
    if mu_sum is None:
        mu_sum = np.zeros(nn_policy.params.state_dim)
        sigma_sq_sum = np.zeros(nn_policy.params.state_dim)

    # Step 3: Training iterations
    for i in range(params.n_iters):
        env_num = np.random.randint(0, len(env_list))
        env = env_list[env_num]
        reward_fn = reward_fns[env_num]

        # Step 3a: Sample deltas
        deltas = []
        for _ in range(params.n_samples):
            # i) Sample delta
            delta = _sample_delta(nn_policy)

            # ii) Construct perturbed policies
            nn_policy_plus = _get_delta_policy(nn_policy, delta, params.delta_std)
            nn_policy_minus = _get_delta_policy(nn_policy, delta, -params.delta_std)

            # iii) Get rollouts
            sardss_plus = get_rollout(env, nn_policy_plus, False, params.timesteps)
            sardss_minus = get_rollout(env, nn_policy_minus, False, params.timesteps)
            num_transitions += (len(sardss_plus) + len(sardss_minus))

            # iv) Estimate cumulative rewards
            r_plus = discounted_reward(sardss_plus, params.gamma, reward_fn)
            r_minus = discounted_reward(sardss_minus, params.gamma, reward_fn)

            # v) Save delta
            deltas.append((delta, r_plus, r_minus))

            # v) Update estimates of normalization parameters
            states = np.array([state for state, _, _, _, _ in sardss_plus + sardss_minus])
            mu_sum += np.sum(states)
            sigma_sq_sum += np.sum(np.square(states))
            n_states += len(states)

        # Step 3b: Sort deltas
        deltas.sort(key=lambda delta: -max(delta[1], delta[2]))
        deltas = deltas[:params.n_top_samples]

        # Step 3c: Compute the sum of the deltas weighted by their reward differences
        delta_sum = [torch.zeros(delta_cur.shape)
                     for delta_cur in deltas[0][0]]
        if nn_policy.use_gpu:
            delta_sum = [delta_cur.cuda() for delta_cur in delta_sum]

        for j in range(params.n_top_samples):
            # i) Unpack values
            delta, r_plus, r_minus = deltas[j]

            # ii) Add delta to the sum
            for k in range(len(delta_sum)):
                delta_sum[k] += (r_plus - r_minus) * delta[k]

        # Step 3d: Compute standard deviation of rewards
        sigma_r = np.std([delta[1] for delta in deltas] +
                         [delta[2] for delta in deltas])

        # Step 3e: Compute step length
        delta_step = [(params.lr * params.delta_std / (params.n_top_samples * sigma_r + 1e-8))
                      * delta_sum_cur for delta_sum_cur in delta_sum]

        # Step 3f: Update policy weights
        nn_policy = _get_delta_policy(nn_policy, delta_step, 1.0)

        # Step 3g: Update normalization parameters
        nn_policy.mu = mu_sum / n_states
        nn_policy.sigma_inv = 1.0 / np.sqrt(sigma_sq_sum / n_states)

        # Step 3h: Logging
        if i % (20 * len(env_list)) == 0:
            cum_rewards = []
            sim_total_steps = 0
            for test_env, test_reward_fn in zip(env_list, reward_fns):
                cum_rew, sim_steps = test_policy(
                    test_env, nn_policy, 20, gamma=params.gamma, max_timesteps=params.timesteps,
                    get_steps=True, reward_fn=test_reward_fn)
                cum_rewards.append(cum_rew)
                sim_total_steps += sim_steps
            if verbose:
                print('Expected rewards at iteration {}: {}'.format(i, cum_rewards))

            # Step 3i: Save best policy
            cum_reward = np.mean(cum_rewards)
            if best_reward <= cum_reward:
                best_policy = nn_policy
                best_reward = cum_reward
            if params.track_best:
                num_transitions += sim_total_steps

        log_info.append([num_transitions, cum_reward])

    # Step 4: Copy new weights and normalization parameters to original policy
    if params.track_best:
        nn_policy = best_policy
    for param, param_orig in zip(nn_policy.parameters(), nn_policy_orig.parameters()):
        param_orig.data.copy_(param.data)
    nn_policy_orig.mu = nn_policy.mu
    nn_policy_orig.sigma_inv = nn_policy.sigma_inv

    return best_policy, np.array(log_info), mu_sum, sigma_sq_sum, n_states


# Construct random perturbations to neural network parameters.
#
# nn_policy: NNPolicy
# return: [torch.tensor] (list of torch tensors that is the same shape as nn_policy.parameters())
def _sample_delta(nn_policy):
    delta = []
    for param in nn_policy.parameters():
        delta_param = torch.normal(torch.zeros(param.shape, dtype=torch.float))
        if nn_policy.use_gpu:
            delta_param = delta_param.cuda()
        delta.append(delta_param)
    return delta


# Construct the policy perturbed by the given delta
#
# nn_policy: NNPolicy
# delta: [torch.tensor] (list of torch tensors that is the same shape as nn_policy.parameters())
# sign: float (should be 1.0 or -1.0, for convenience)
# return: NNPolicy
def _get_delta_policy(nn_policy, delta, sign):
    # Step 1: Construct the perturbed policy
    nn_policy_delta = NNPolicy(nn_policy.params, use_gpu=nn_policy.use_gpu)

    # Step 2: Set normalization of the perturbed policy
    nn_policy_delta.mu = nn_policy.mu
    nn_policy_delta.sigma_inv = nn_policy.sigma_inv

    # Step 3: Set the weights of the perturbed policy
    for param, param_delta, delta_cur in zip(nn_policy.parameters(), nn_policy_delta.parameters(),
                                             delta):
        param_delta.data.copy_(param.data + sign * delta_cur)

    return nn_policy_delta
