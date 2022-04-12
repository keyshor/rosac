from hybrid_gym.rl.sac.sac import MySAC
from hybrid_gym.eval import mcts_eval, random_selector_eval

import numpy as np
import random
import time


class MaSAC:

    def __init__(self, automaton, max_ep_len, time_limits, max_jumps,
                 sac_kwargs, reward_fns, epsilon=0.15, verbose=False,
                 use_gpu=False):
        self.automaton = automaton
        self.max_ep_len = max_ep_len
        self.time_limits = time_limits
        self.max_jumps = max_jumps
        self.reward_fns = reward_fns
        self.epsilon = epsilon
        self.verbose = verbose

        # Create agent trainers
        self.trainers = {m: MySAC(automaton.observation_space, automaton.action_space,
                                  **sac_kwargs) for m in automaton.modes}
        if use_gpu:
            for m, trainer in self.trainers.items():
                trainer.gpu()
        self.controllers = {m: trainer.get_policy(deterministic=False)
                            for m, trainer in self.trainers.items()}
        self.det_controllers = {m: trainer.get_policy() for m, trainer in self.trainers.items()}

        # Update the value fns used in reward_fns
        value_fns = {m: [trainer.get_policy(deterministic=False, use_target=True)]
                     for m, trainer in self.trainers.items()}
        for rfn in self.reward_fns.values():
            rfn.update([], value_fns, copy_value_fns=False)

    def train(self, max_steps):

        log_info = []
        abs_start_time = time.time()

        mname = random.choice(list(self.automaton.modes))
        mode = self.automaton.modes[mname]
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
            if self.trainers[mname].replay_buffer.size > self.trainers[mname].start_steps:
                action = self.controllers[mname].get_action(obs)
            else:
                action = self.automaton.action_space.sample()

            # environment step
            new_state = mode.step(state, action)
            new_obs = mode.observe(new_state)
            rew = mode.reward(state, action, new_state)
            info, transition = self.compute_info(new_state, self.automaton.transitions[mname])
            unsafe = not mode.is_safe(new_state)

            self.trainers[mname].replay_buffer.store(
                obs, action, rew, new_obs, unsafe or info['is_success'], info)

            corrected_r = self.reward_fns[mname].obs_reward(
                obs, action, new_obs, rew, info)
            episode_reward += corrected_r
            episode_step += 1
            total_step += 1

            obs = new_obs
            state = new_state

            if info['is_success'] and not unsafe:
                # epsilon greedy
                if np.random.uniform() <= self.epsilon:
                    mname = random.choice(list(self.automaton.modes))
                else:
                    _, mname = self.reward_fns[mname].compute_value(info['jump_obs'])
                mode = self.automaton.modes[mname]
                state = transition.jump(mname, state)
                obs = mode.observe(state)

            # update all trainers
            if total_step % len(self.automaton.modes) == 0:
                for m, agent in self.trainers.items():
                    if (agent.replay_buffer.size >= agent.update_after and
                            train_step % agent.update_every == 0):
                        for j in range(agent.update_every):
                            batch = agent.replay_buffer.sample_batch(
                                agent.batch_size, reward_fn=self.reward_fns[m])
                            agent.update(data=batch)

                # increment global step counter
                train_step += 1

            if unsafe or episode_step > self.max_ep_len:
                mname = random.choice(list(self.automaton.modes))
                mode = self.automaton.modes[mname]
                state = mode.end_to_end_reset()
                obs = mode.observe(state)

                if self.verbose:
                    print('Reward at episode {}: {}'.format(num_episodes, episode_reward))

                episode_reward = 0.
                episode_step = 0
                num_episodes += 1

            if total_step % 20000 == 0 and total_step != 0:
                mcts_prob, mcts_avg_jmps, _ = mcts_eval(
                    self.automaton, self.det_controllers, self.time_limits,
                    max_jumps=self.max_jumps, mcts_rollouts=1000, eval_rollouts=100)
                rs_prob, avg_jmps, _ = random_selector_eval(
                    self.automaton, self.det_controllers, self.time_limits,
                    max_jumps=self.max_jumps, eval_rollouts=100)
                time_taken = time.time() - abs_start_time
                log_info.append([total_step, time_taken, avg_jmps,
                                mcts_avg_jmps, rs_prob, mcts_prob])
                print('MCTS: Avg Jumps = {} | Prob = {}'.format(mcts_avg_jmps, mcts_prob))
                print('Random: Avg Jumps = {} | Prob = {}'.format(avg_jmps, rs_prob))

            # saves final episode reward for plotting training curve later
            if total_step > max_steps:
                print('Finished training.')
                break

        return np.array(log_info)

    def compute_info(self, state, transitions):
        is_success = False
        jump_obs = None
        transition = None
        for t in transitions:
            if t.guard(state):
                is_success = True
                transition = t
                jump_obs = []
                for target in t.targets:
                    jump_state = t.jump(target, state)
                    jump_obs.append(
                        (target, self.automaton.modes[target].observe(jump_state)))
                break
        return dict(is_success=is_success, jump_obs=jump_obs), transition
