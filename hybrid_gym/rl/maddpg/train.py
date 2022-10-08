import tensorflow as tf
import numpy as np
import time

import tensorflow.contrib.layers as layers
from hybrid_gym.model import Controller
from hybrid_gym.eval import random_selector_eval, mcts_eval
from hybrid_gym.util.wrappers import AdvEnv
from hybrid_gym.rl.maddpg.common import tf_util as U
from hybrid_gym.rl.maddpg.trainer.maddpg import MADDPGAgentTrainer


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64,
              activation_fn=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=activation_fn)
        return out


def get_trainers(env, obs_shape, arglist, sess):
    agent = MADDPGAgentTrainer("agent", mlp_model, obs_shape, env.action_space[0],
                               arglist, sess, num_modes=env.action_space[1].n)
    adv = MADDPGAgentTrainer("adv", mlp_model, obs_shape, env.action_space[0],
                             arglist, sess, adv=True, num_modes=env.action_space[1].n)
    return agent, adv


class MADDPGParams:

    def __init__(self, max_episode_len, num_episodes, lr=3e-4,
                 gamma=0.95, batch_size=256, num_units=64, log_rate=500):
        self.max_episode_len = max_episode_len
        self.num_episodes = num_episodes
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_units = num_units
        self.log_rate = log_rate


class MADDPG:

    def __init__(self, automaton, params, bonus=25.):
        self.params = params
        self.graph = tf.Graph()
        self.session = U.single_threaded_session(self.graph)

        self.mode_list = [m for m in automaton.modes]
        self.env = AdvEnv(automaton, self.mode_list, bonus=bonus)
        self.automaton = automaton

        with self.graph.as_default():

            # Create agent trainers
            self.obs_shape = self.env.observation_space.shape
            self.agent, self.adv = get_trainers(self.env, self.obs_shape, self.params, self.session)

            # Initialize
            U.initialize_once(self.session)

    def train(self, time_limits, max_jumps, max_total_steps):

        episode_rewards = [0.0]  # rewards for agent
        obs = self.env.reset()
        episode_step = 0
        train_step = 0
        abs_start_time = time.time()
        log_info = []
        steps_taken = 0
        # t_start = time.time()
        # saver = tf.train.Saver()

        print('Starting iterations...')
        while True:

            # get action
            action = self.agent.action(obs)
            adv_action = self.adv.action(obs)
            adv_action = min(len(self.mode_list), adv_action)

            # environment step
            new_obs, rew, done, info = self.env.step(action, adv_action)
            episode_step += 1
            steps_taken += 1
            terminal = done or (episode_step > self.params.max_episode_len)

            # collect experience
            self.agent.experience(obs, action, rew, new_obs, terminal)
            self.adv.experience(obs, adv_action, -rew, new_obs, terminal)
            obs = new_obs

            episode_rewards[-1] += rew

            # update all trainers, if not in display or benchmark mode
            self.agent.preupdate()
            self.adv.preupdate()
            self.agent.update(self.agent, self.adv, train_step)
            self.adv.update(self.agent, self.adv, train_step)

            # increment global step counter
            train_step += 1

            # estimate current policies and display stats
            if terminal and (len(episode_rewards) % self.params.log_rate == 0):
                mode_controllers = self.get_controllers()
                mcts_prob, mcts_avg_jmps, _ = mcts_eval(
                    self.automaton, mode_controllers, time_limits, max_jumps=max_jumps,
                    mcts_rollouts=1000, eval_rollouts=100)
                rs_prob, avg_jmps, _ = random_selector_eval(
                    self.automaton, mode_controllers, time_limits, max_jumps=max_jumps,
                    eval_rollouts=100)
                time_taken = time.time() - abs_start_time
                log_info.append([train_step, time_taken, avg_jmps,
                                mcts_avg_jmps, rs_prob, mcts_prob])

            if terminal:
                obs = self.env.reset()
                episode_step = 0
                print('Reward at episode {}: {}'.format(len(episode_rewards), episode_rewards[-1]))
                episode_rewards.append(0)

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > self.params.num_episodes or steps_taken >= max_total_steps:
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

        return log_info

    def get_controllers(self, deterministic=True):
        return {self.mode_list[i]: MADDPGController(
            self.agent, i, len(self.mode_list), deterministic)
            for i in range(len(self.mode_list))}


class MADDPGController(Controller):

    def __init__(self, agent, mode, num_modes, deterministic=True):
        self.agent = agent
        self.deterministic = deterministic
        self.mode_enc = np.zeros((num_modes,))
        self.mode_enc[mode] = 1.

    def get_action(self, obs):
        obs = np.concatenate([self.mode_enc, obs])
        if self.deterministic:
            return self.agent.det_action(obs)
        else:
            return self.agent.action(obs)

    def save(self, name, path):
        pass
