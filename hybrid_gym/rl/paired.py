import time
import torch
import numpy as np
import torch.nn as nn

from torch.distributions import Categorical
from hybrid_gym.rl.ppo.multi_agent import Model, Trainer
from hybrid_gym.rl.ddpg.ddpg import optimizer_to
from hybrid_gym.eval import mcts_eval, random_selector_eval


class Adversary:
    '''
    Adversary with REINFORCE updates
    '''

    def __init__(self, automaton, max_jumps, gpu_device, learning_rate=1e-3):
        self.mode_list = [mname for mname in automaton.modes]
        self.logits = nn.Parameter(torch.zeros(
            (max_jumps, len(self.mode_list)), dtype=torch.float32))
        self.gpu_device = gpu_device
        self.optimizer = torch.optim.Adam([self.logits], lr=learning_rate)
        self.device_used = 'cpu'

    def sample_modes(self):
        dist = Categorical(logits=self.logits)
        sample = dist.sample().squeeze().cpu().numpy()
        return [self.mode_list[mnum] for mnum in sample], sample

    def log_prob(self, sample):
        dist = Categorical(logits=self.logits)
        return dist.log_prob(torch.tensor(sample, device=self.device_used)).sum(axis=-1)

    def gpu(self):
        self.logits = self.logits.to(self.gpu_device)
        optimizer_to(self.optimizer, self.gpu_device)
        self.device_used = self.gpu_device

    def cpu(self):
        self.logits = self.logits.to('cpu')
        optimizer_to(self.optimizer, 'cpu')
        self.device_used = 'cpu'

    def update(self, samples, regrets):
        log_probs = self.log_prob(np.array(samples))
        loss = -torch.dot(log_probs, torch.tensor(regrets,
                          dtype=torch.float32, device=self.device_used))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Paired:

    def __init__(self, automaton, hidden_dim,
                 gpu_device, trainer_config, time_limits,
                 log_std_max=2, log_std_min=-20, adv_lr=1e-3,
                 adversary_batch_size=1, adversary_updates=1,
                 max_jumps=100, use_gpu=False, eval_every=25,
                 ):
        # Inititalize models
        self.protagonist = Model(automaton, hidden_dim, gpu_device,
                                 log_std_max=log_std_max, log_std_min=log_std_min)
        self.antagonist = Model(automaton, hidden_dim, gpu_device,
                                log_std_max=log_std_max, log_std_min=log_std_min)
        self.adversary = Adversary(automaton, max_jumps, gpu_device, learning_rate=adv_lr)

        # move to gpu if necessary
        device = "cpu"
        if use_gpu:
            self.protagonist.gpu()
            self.antagonist.gpu()
            self.adversary.gpu()
            device = gpu_device

        # Initialize trainer objects for agents
        self.p_trainer = Trainer(model=self.protagonist, **trainer_config,
                                 training_device=device)
        self.a_trainer = Trainer(model=self.antagonist, **trainer_config,
                                 training_device=device)

        # Initialize controllers
        self.det_controllers = self.p_trainer.get_controllers()

        # store automaton
        self.automaton = automaton

        # hyperparams
        self.adv_batch_size = adversary_batch_size
        self.adv_updates = adversary_updates
        self.eval_every = eval_every
        self.time_limits = time_limits
        self.max_jumps = max_jumps

    def train(self, num_iter):

        steps = 0
        start_time = time.time()
        log_info = []

        adv_samples = []
        adv_regrets = []

        for i in range(num_iter):

            modes, mode_nums = self.adversary.sample_modes()
            p_sample, max_p_reward = self.p_trainer.sample(self.automaton, modes)
            a_sample, max_a_reward = self.a_trainer.sample(self.automaton, modes)

            self.p_trainer.adjust_rewards(p_sample, max_a_reward)
            self.a_trainer.adjust_rewards(a_sample, max_p_reward)

            adv_samples.append(mode_nums)
            adv_regrets.append(self.compute_regret(p_sample))
            steps += len(p_sample['done']) + len(a_sample['done'])

            # train agents
            self.p_trainer.train(p_sample)
            self.a_trainer.train(a_sample)

            # train adversary
            if len(adv_samples) >= self.adv_batch_size:
                for _ in range(self.adv_updates):
                    self.adversary.update(adv_samples, adv_regrets)
                adv_regrets = []
                adv_samples = []

            # evaluation
            if i % self.eval_every == 0 and i > 0:
                mcts_prob, mcts_avg_jmps, _ = mcts_eval(
                    self.automaton, self.det_controllers, self.time_limits,
                    max_jumps=self.max_jumps, mcts_rollouts=1000, eval_rollouts=100)
                rs_prob, avg_jmps, _ = random_selector_eval(
                    self.automaton, self.det_controllers, self.time_limits,
                    max_jumps=self.max_jumps, eval_rollouts=100)
                time_taken = time.time() - start_time
                log_info.append([steps, time_taken, avg_jmps,
                                mcts_avg_jmps, rs_prob, mcts_prob])
                print('MCTS: Avg Jumps = {} | Prob = {}'.format(mcts_avg_jmps, mcts_prob))
                print('Random: Avg Jumps = {} | Prob = {}'.format(avg_jmps, rs_prob))

        return np.array(log_info)

    def compute_regret(self, sample):
        num_episodes = np.sum(sample['done'])
        if not sample['done'][-1]:
            num_episodes += 1
        return -np.sum(sample['rewards']) / num_episodes
