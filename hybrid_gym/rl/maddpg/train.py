import tensorflow as tf
import numpy as np
import random
import gym

import tensorflow.contrib.layers as layers
from hybrid_gym.rl.maddpg.common import tf_util as U
from hybrid_gym.rl.maddpg.trainer.maddpg import MADDPGAgentTrainer
from hybrid_gym.model import Controller
from hybrid_gym.eval import mcts_eval, random_selector_eval


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None,
              activation_fn=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=activation_fn)
        return out


def get_trainers(automaton, arglist, sess):
    trainers = []
    adversaries = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    obs_shape = automaton.observation_space.shape
    adv_act_space = gym.spaces.Discrete(len(automaton.modes))
    mode2int = {}
    mode_num = 0
    for mname in automaton.modes:
        trainers.append(trainer(mname, model, obs_shape,
                                automaton.action_space, arglist, sess))
        adversaries.append(trainer(mname + '_adv', model, obs_shape,
                                   adv_act_space, arglist, sess, adversary=True))
        mode2int[mname] = mode_num
        mode_num += 1
    return trainers, adversaries, mode2int


class MADDPGParams:

    def __init__(self, max_episode_len, num_train_steps, lr=3e-4,
                 gamma=0.95, batch_size=1024, num_units=64):
        self.max_episode_len = max_episode_len
        self.num_train_steps = num_train_steps
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_units = num_units


class MADDPG:

    def __init__(self, automaton, params):
        self.automaton = automaton
        self.params = params
        self.graph = tf.Graph()
        self.session = U.single_threaded_session(self.graph)

        with self.graph.as_default():

            # Create agent trainers
            self.trainers, self.adversaries, self.mode2int = get_trainers(
                self.automaton, self.params, self.session)

            # Initialize
            U.initialize_once(self.session)

    def train(self, time_limits, max_jumps):

        log_info = []

        m_num = random.choice(list(range(len(self.trainers))))
        mode = self.automaton.modes[self.trainers[m_num].name]
        state = mode.end_to_end_reset()
        obs = mode.observe(state)
        episode_step = 0
        train_step = 0
        total_step = 0
        num_episodes = 0
        episode_reward = 0.

        print('Starting MADDPG Training...')
        while True:

            # get action
            action = self.trainers[m_num].action(obs)

            # environment step
            # print('{}: {}'.format(mode.name, state))
            new_state = mode.step(state, action)
            new_obs = mode.observe(new_state)
            rew = mode.reward(state, action, new_state)
            transition = None
            done = False

            for t in self.automaton.transitions[mode.name]:
                if t.guard(new_state):
                    transition = t

            if not mode.is_safe(new_state):
                done = True

            success = (transition is not None)

            self.trainers[m_num].experience(obs, action, rew, new_obs, done,
                                            success)

            episode_step += 1
            total_step += 1
            episode_reward += (rew + float(success) * 50.)
            obs = new_obs
            state = new_state

            if transition is not None and not done:
                adv_action = self.adversaries[m_num].action(obs)
                new_obs = [None] * len(adv_action)
                for target in transition.targets:
                    new_state = transition.jump(target, state)
                    new_mode = self.automaton.modes[target]
                    new_obs[self.mode2int[target]] = new_mode.observe(new_state)

                self.adversaries[m_num].experience(obs, adv_action, 0., new_obs,
                                                   True, True)
                m_num = np.random.choice(list(range(len(adv_action))),
                                         p=adv_action)
                mode = self.automaton.modes[self.trainers[m_num].name]
                obs = new_obs[m_num]
                state = transition.jump(mode.name, state)

            # update all trainers, if not in display or benchmark mode
            for agent in self.trainers + self.adversaries:
                agent.preupdate()
            for agent, adversary in zip(self.trainers, self.adversaries):
                agent.update(adversary, train_step)
                adversary.update(self.trainers, train_step)

            # increment global step counter
            train_step += 1

            if done or episode_step > self.params.max_episode_len:
                m_num = random.choice(list(range(len(self.trainers))))
                mode = self.automaton.modes[self.trainers[m_num].name]
                state = mode.end_to_end_reset()
                obs = mode.observe(state)

                print('Reward at episode {}: {}'.format(
                    num_episodes, episode_reward))
                episode_reward = 0.
                episode_step = 0
                num_episodes += 1

            if total_step % 20000 == 0 and total_step != 0:
                mode_controllers = self.get_policies(deterministic=True)
                mcts_prob, mcts_avg_jmps, _ = mcts_eval(
                    self.automaton, mode_controllers, time_limits, max_jumps=max_jumps,
                    mcts_rollouts=1000, eval_rollouts=100)
                rs_prob, avg_jmps, _ = random_selector_eval(
                    self.automaton, mode_controllers, time_limits, max_jumps=max_jumps,
                    eval_rollouts=100)
                log_info.append([total_step, avg_jmps, mcts_avg_jmps, rs_prob, mcts_prob])

            # saves final episode reward for plotting training curve later
            if total_step > self.params.num_train_steps:
                print('Finished training.')
                break

        return np.array(log_info)

    def get_policies(self, deterministic=False, copy=True):
        return {agent.name: AgentPolicy(agent, deterministic, copy)
                for agent in self.trainers}


class AgentPolicy(Controller):

    def __init__(self, trainer, deterministic=False, copy=False):
        self.trainer = trainer
        self.deterministic = deterministic
        self.copy = copy

    def get_action(self, obs):
        if self.deterministic:
            return self.trainer.deterministic_action(obs, self.copy)
        else:
            return self.trainer.action(obs, self.copy)

    def save(self, name, path):
        pass
