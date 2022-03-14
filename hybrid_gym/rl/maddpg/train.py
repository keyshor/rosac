import tensorflow as tf
import time

import tensorflow.contrib.layers as layers
from spectrl.rl.maddpg.common import tf_util as U
from spectrl.rl.maddpg.trainer.maddpg import MADDPGAgentTrainer
from spectrl.util.rl import MultiAgentPolicy, test_policy_mutli


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None,
              activation_fn=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=activation_fn)
        return out


def get_trainers(env, obs_shape_n, arglist, sess):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(env.n):
        trainers.append(trainer("agent_%d" % i, model, obs_shape_n,
                                env.action_space, i, arglist, sess))
    return trainers


class MADDPGParams:

    def __init__(self, max_episode_len, num_episodes, lr=3e-4,
                 gamma=0.95, batch_size=1024, num_units=64, log_rate=30):
        self.max_episode_len = max_episode_len
        self.num_episodes = num_episodes
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_units = num_units
        self.log_rate = log_rate


class MADDPG:

    def __init__(self, multi_env, params):
        self.env = multi_env
        self.params = params
        self.graph = tf.Graph()
        self.session = U.single_threaded_session(self.graph)

        with self.graph.as_default():

            # Create agent trainers
            self.obs_shape_n = [self.env.observation_space[i].shape for i in range(self.env.n)]
            self.trainers = get_trainers(self.env, self.obs_shape_n, self.params, self.session)

            # Initialize
            U.initialize_once(self.session)

        self.multi_policy = MultiAgentPolicy(self.get_policies(copy=False))

    def train(self, main_agent=None):

        # store best set of policies wrt expected reward of the main agent
        if main_agent is not None:
            best_reward = -1e9

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(self.env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        obs_n = self.env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        # saver = tf.train.Saver()

        print('Starting iterations...')
        while True:

            # get action
            action_n = [agent.action(obs) for agent, obs in zip(self.trainers, obs_n)]

            # environment step
            new_obs_n, rew_n, done, info_n = self.env.step(action_n)
            episode_step += 1
            terminal = (episode_step > self.params.max_episode_len)

            # collect experience
            for i, agent in enumerate(self.trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i],
                                 new_obs_n[i], done, terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            # update all trainers, if not in display or benchmark mode
            for agent in self.trainers:
                agent.preupdate()
            for agent in self.trainers:
                agent.update(self.trainers, train_step)

            # increment global step counter
            train_step += 1

            # estimate current policies and display stats
            if terminal and (len(episode_rewards) % self.params.log_rate == 0):
                test_rewards, _ = test_policy_mutli(
                    self.env, self.multi_policy, 20, self.params.gamma,
                    max_timesteps=self.params.max_episode_len)
                print("steps: {}, episodes: {}, mean agent reward: {}, time: {}".format(
                    train_step, len(episode_rewards), test_rewards.tolist(),
                    round(time.time()-t_start, 3)))
                # t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(sum(test_rewards))
                final_ep_ag_rewards.append(test_rewards)

                if (main_agent is not None) and \
                        (len(episode_rewards) > (self.params.num_episodes/2)):
                    if best_reward < test_rewards[main_agent]:
                        best_reward = test_rewards[main_agent]
                        for agent in self.trainers:
                            agent.copy_update_p()

            if done or terminal:
                obs_n = self.env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > self.params.num_episodes:
                if main_agent is not None:
                    for agent in self.trainers:
                        agent.copy_update_p()
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

        return train_step

    def get_policies(self, deterministic=False, copy=True):
        return [AgentPolicy(agent, deterministic, copy) for agent in self.trainers]


class AgentPolicy:

    def __init__(self, trainer, deterministic=False, copy=False):
        self.trainer = trainer
        self.deterministic = deterministic
        self.copy = copy

    def get_action(self, obs):
        if self.deterministic:
            return self.trainer.deterministic_action(obs, self.copy)
        else:
            return self.trainer.action(obs, self.copy)


# example usage
if __name__ == '__main__':
    from spectrl.envs.particles import MultiParticleEnv
    from spectrl.util.rl import get_rollout

    start_pos_low = [[-0.1, -0.1], [-0.1, 9.9]]
    start_pos_high = [[0.1, 0.1], [0.1, 10.1]]
    goals = [[10., 0.], [10., 10.]]
    env = MultiParticleEnv(start_pos_low, start_pos_high, goals=goals,
                           sum_rewards=False, max_timesteps=30)
    trainer = MADDPG(env, MADDPGParams(30, 20000, num_units=64, lr=3e-4))
    trainer.train()
    policy = MultiAgentPolicy(trainer.get_policies())
    get_rollout(env, policy, True)
