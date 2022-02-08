'''
CounterExample Guided Reinforcement Learning
'''

from hybrid_gym import HybridAutomaton, Mode, Controller
from hybrid_gym.train.single_mode import (make_sb3_model_init_check, train_sb3,
                                          make_ars_model, parallel_ars, learn_ddpg_model,
                                          make_ddpg_model, make_sac_model, parallel_sac)
from hybrid_gym.synthesis.abstractions import AbstractState
from hybrid_gym.synthesis.ice import synthesize
from hybrid_gym.util.wrappers import Sb3CtrlWrapper
from hybrid_gym.falsification.single_mode import falsify
from hybrid_gym.train.reward_funcs import RewardFunc
from hybrid_gym.eval import mcts_eval, random_selector_eval
from typing import List, Dict, Any, Iterable, Callable, Optional, Tuple
from multiprocessing import Process, Queue
import multiprocessing as mp

import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt


class ResetFunc:
    '''
    Reset function used to sample start states in training.
    '''
    mode: Mode
    states: List[Any]
    prob: float
    full_reset: bool

    def __init__(self, mode: Mode, states: Iterable[Any] = [], prob: float = 0.75,
                 full_reset: bool = False) -> None:
        self.mode = mode
        self.states = list(states)
        self.prob = prob
        self.full_reset = full_reset

    def __call__(self):
        if np.random.binomial(1, self.prob) and len(self.states) > 0:
            return random.choice(self.states)
        elif self.full_reset:
            return self.mode.reset()
        else:
            return self.mode.end_to_end_reset()

    def add_states(self, states: Iterable[Any]) -> None:
        self.states.extend(states)

    def make_serializable(self):
        self.mode = self.mode.name
        return self

    def recover_after_serialization(self, automaton):
        self.mode = automaton.modes[self.mode]
        return self


def cegrl(automaton: HybridAutomaton,
          pre: Dict[str, AbstractState],
          time_limits: Dict[str, int],
          mode_groups: List[List[Mode]] = [],
          reward_funcs: Optional[Dict[str, Optional[RewardFunc]]] = None,
          max_jumps: int = 10,
          algo_name: str = 'td3',
          num_iter: int = 20,
          ensemble: int = 1,
          dagger: bool = False,
          use_gpu: bool = False,
          full_reset: bool = False,
          env_name: Optional[str] = None,
          num_synth_iter: int = 10,
          n_synth_samples: int = 50,
          abstract_synth_samples: int = 0,
          inductive_ce: bool = False,
          num_falsification_iter: int = 200,
          num_falsification_samples: int = 20,
          num_falsification_top_samples: int = 10,
          falsify_func: Optional[Dict[str, Callable[[List[Any]], float]]] = None,
          print_debug: bool = False,
          save_path: str = '.',
          plot_synthesized_regions: bool = False,
          **kwargs
          ) -> Tuple[Dict[str, Controller], np.ndarray]:
    '''
    Train policies for all modes

    * Set 'full_reset' to True to use reset() instead of end_to_end_reset()
        as default sampling method.
    * Set 'num_synth_iter' to 0 to not use synthesis (counterexample guided learning).
        Optionally set 'dagger' to True to use dataset aggregation.
    * When synthesis is enabled, optionally provide 'falsify_func' to be used
        CE method to find bad states.
    * 'abstract_synth_samples' denotes the number of states sampled from the synthesized
        abstract states (using the corresponding sampling function of the domain).
    '''

    log_info = []
    steps_taken = 0
    cond_prob_file = open(os.path.join(save_path, 'cond_probs.txt'), 'w')

    mp.set_start_method('spawn')

    # define reset functions
    reset_funcs = {name: ResetFunc(mode, full_reset=full_reset)
                   for (name, mode) in automaton.modes.items()}
    if reward_funcs is None:
        reward_funcs = {name: None for name in automaton.modes}

    # Add each mode into its own group if no grouping is given
    if len(mode_groups) == 0:
        mode_groups = [[mode] for _, mode in automaton.modes.items()]
    group_names = [[mode.name for mode in modes] for modes in mode_groups]

    # map modes to their group numbers
    group_map = {}
    for g in range(len(mode_groups)):
        for name in group_names[g]:
            group_map[name] = g

    # all necessary information about all groups
    group_info = [[(mode, automaton.transitions[mode.name],
                    reset_funcs[mode.name], reward_funcs[mode.name])
                   for mode in modes] for modes in mode_groups]
    serializable_info = [[(mode.name, reset_fn, reward_fn)
                         for mode, _, reset_fn, reward_fn in group] for group in group_info]

    # ensemble models for each group
    ens_models: List[List[Any]] = []
    ens_controllers: List[List[Controller]] = []

    # create ensemble models and controllers for each group
    if algo_name == 'ars':
        ens_models = [[make_ars_model(**kwargs['ars_kwargs'])
                       for _ in range(ensemble)] for _ in mode_groups]
        ens_controllers = [[model.nn_policy for model in models] for models in ens_models]
    elif algo_name == 'my_ddpg':
        ens_models = [[make_ddpg_model(**kwargs['ddpg_kwargs'], use_gpu=use_gpu)
                       for _ in range(ensemble)] for _ in mode_groups]
        ens_controllers = [[model.get_policy() for model in models] for models in ens_models]
    elif algo_name == 'my_sac':
        ens_models = [[make_sac_model(
            automaton.observation_space, automaton.action_space, **kwargs['sac_kwargs'])
            for _ in range(ensemble)]
            for _ in mode_groups]
        ens_controllers = [[model.get_policy() for model in models] for models in ens_models]
        ens_stoch_controllers = [[model.get_policy(deterministic=False) for model in models]
                                 for models in ens_models]
    else:
        ens_models = [[make_sb3_model_init_check(
            group_info[g],
            algo_name=algo_name,
            max_episode_steps=time_limits[group_names[g][0]],
            **kwargs['sb3_kwargs']) for _ in range(ensemble)]
            for g in range(len(mode_groups))]
        ens_controllers = [[Sb3CtrlWrapper(model) for model in models] for models in ens_models]

    for i in range(num_iter):
        start_time = time.time()
        print('\nStarting to train individual controllers in iteration {} ...'.format(i))

        # parallelize learning, initialize list of queues to retrieve trained models
        if algo_name == 'ars' or algo_name == 'my_sac':
            ret_queues: List[List[Queue]] = [[] for group in mode_groups]
            req_queues: List[List[Queue]] = [[] for group in mode_groups]
            processes: List[List[Process]] = [[] for group in mode_groups]

        # train agents
        for g in range(len(mode_groups)):
            if print_debug:
                print('\nTraining controllers for modes {}'.format(group_names[g]))

            for _, reward_fn, reset_fn in serializable_info[g]:
                if reward_fn is not None:
                    reward_fn.make_serializable()
                if reset_fn is not None:
                    reset_fn.make_serializable()

            for e in range(ensemble):

                # ARS and SAC support parallelization
                if algo_name == 'ars' or algo_name == 'my_sac':

                    # Initialize return and request queue for models[g][e]
                    ret_queues[g].append(Queue())
                    req_queues[g].append(Queue())

                    # move everything to CPU since multiprocessing library
                    # doesn't handle GPU resources
                    if use_gpu:
                        ens_models[g][e].cpu()

                    # set the correspondning training functions
                    if algo_name == 'ars':
                        processes[g].append(Process(target=parallel_ars, args=(
                            ens_models[g][e], env_name, serializable_info[g], save_path,
                            ret_queues[g][e], req_queues[g][e],
                            print_debug, use_gpu)))
                    else:
                        processes[g].append(Process(target=parallel_sac, args=(
                            ens_models[g][e], env_name, serializable_info[g],
                            ret_queues[g][e], req_queues[g][e],
                            print_debug, (i > 0), use_gpu)))

                    # start the training process
                    processes[g][e].start()

                # sequential training for ddpg and stable_baselines
                elif algo_name == 'my_ddpg':
                    steps_taken += learn_ddpg_model(ens_models[g][e], list(group_info[g]))
                else:
                    steps_taken += train_sb3(
                        model=ens_models[g][e],
                        raw_mode_info=group_info[g],
                        algo_name=algo_name,
                        save_path=save_path,
                        max_episode_steps=time_limits[group_names[g][0]],
                        **kwargs['sb3_train_kwargs'],
                    )

            for _, reward_fn, reset_fn in serializable_info[g]:
                if reward_fn is not None:
                    reward_fn.recover_after_serialization(automaton)
                if reset_fn is not None:
                    reset_fn.recover_after_serialization(automaton)

        # retrieve new controllers
        if algo_name == 'ars' or algo_name == 'my_sac':
            for g in range(len(mode_groups)):
                for e in range(ensemble):
                    while True:
                        try:
                            req_queues[g][e].put(1)
                            ens_models[g][e], steps = ret_queues[g][e].get()
                            if use_gpu:
                                ens_models[g][e].gpu()
                            break
                        except RuntimeError:
                            print('Runtime Error occured while retrieving policy! Retrying...')
                            continue

                    # stop the learning process and join
                    req_queues[g][e].put(None)
                    processes[g][e].join()

                    # set the new controllers
                    ens_controllers[g][e] = ens_models[g][e].get_policy()
                    if algo_name == 'my_sac':
                        ens_stoch_controllers[g][e] = ens_models[g][e].get_policy(
                            deterministic=False)

                    steps_taken += steps

        print('Completed training individual controllers in {} mins'.format(
            (time.time() - start_time) / 60))

        # evaluating controllers
        start_time = time.time()
        print('\nEvaluating controllers in iteration {} ...'.format(i))

        # TODO: Pick the best controller for each mode
        best_prob = best_mcts_prob = best_jmps = best_mcts_jmps = 0.
        total_eval_steps = 0
        all_collected_states: Dict[str, List] = {m: [] for m in automaton.modes}
        for e in range(ensemble):
            mode_controllers = {name: ens_controllers[g][e] for name, g in group_map.items()}
            mcts_prob, mcts_avg_jmps, _ = mcts_eval(
                automaton, mode_controllers, time_limits, max_jumps=max_jumps, mcts_rollouts=1000,
                eval_rollouts=100)
            rs_prob, avg_jmps, collected_states, eval_steps = random_selector_eval(
                automaton, mode_controllers, time_limits, max_jumps=max_jumps, eval_rollouts=100,
                return_steps=True, conditional_prob_log=cond_prob_file)
            best_prob = max(best_prob, rs_prob)
            best_jmps = max(best_jmps, avg_jmps)
            best_mcts_prob = max(best_mcts_prob, mcts_prob)
            best_mcts_jmps = max(best_mcts_jmps, mcts_avg_jmps)
            total_eval_steps += eval_steps
            for m in automaton.modes:
                all_collected_states[m].extend(collected_states[m])

        log_info.append([steps_taken, best_jmps, best_mcts_jmps, best_prob, best_mcts_prob])

        # probabilistic policies for exploration
        if algo_name == 'my_sac':
            all_collected_states = {m: [] for m in automaton.modes}
            total_eval_steps = 0
            for e in range(ensemble):
                stochastic_mode_controllers = {
                    name: ens_stoch_controllers[g][e] for name, g in group_map.items()}
                _, _, collected_states, eval_steps = random_selector_eval(
                    automaton, stochastic_mode_controllers, time_limits, max_jumps=max_jumps,
                    eval_rollouts=100, return_steps=True)
                total_eval_steps += eval_steps
                for m in automaton.modes:
                    all_collected_states[m].extend(collected_states[m])

        print('Completed evaluation of controllers in {} secs'.format(time.time() - start_time))

        # synthesis
        if num_synth_iter > 0 and ensemble == 1:
            start_time = time.time()
            print('\nRunning synthesis in iteration {} ...'.format(i))

            pre_copy = {name: astate.copy() for name, astate in pre.items()}
            ces, steps = synthesize(automaton, mode_controllers, pre_copy, time_limits,
                                    num_synth_iter, n_synth_samples, abstract_synth_samples,
                                    print_debug, inductive_ce=inductive_ce)
            steps_taken += steps

            if falsify_func is not None:
                # use falsification to identify bad states
                for (m, pre_m) in pre_copy.items():
                    bad_states, steps = falsify(automaton.modes[m], automaton.transitions[m],
                                                mode_controllers[m], pre_copy[m], falsify_func[m],
                                                time_limits[m], num_falsification_iter,
                                                num_falsification_samples,
                                                num_falsification_top_samples)
                    steps_taken += steps
                    reset_funcs[m].add_states(bad_states)
            else:
                # add counterexamples to reset function
                for ce in ces:
                    reset_funcs[ce.m].add_states([ce.s])

            print('Completed synthesis in {} secs'.format(time.time() - start_time))

        # dataset aggregation using random selector
        elif dagger:
            steps_taken += total_eval_steps
            for m in all_collected_states:
                reset_funcs[m].add_states([s[0] for s in all_collected_states[m]])

        # change reward function based on collected states
        start_time = time.time()
        print('\nUpdatign rewards in iteration {} ...'.format(i))
        mode_controller_list = {name: ens_controllers[g] for name, g in group_map.items()}
        for m, mode_reward_fn in reward_funcs.items():
            if mode_reward_fn is not None:
                steps_taken += mode_reward_fn.update(all_collected_states[m], mode_controller_list)
        print('Completed updating rewards in {} secs'.format(time.time() - start_time))

        if plot_synthesized_regions:
            for (name, rf) in reset_funcs.items():
                fig, ax = plt.subplots()
                automaton.modes[name].plot_state_iterable(ax=ax, sts=[rf() for _ in range(200)])
                ax.set_title(f'start_{name}_iter{i}')
                ax.set_aspect('equal')
                fig.savefig(os.path.join(save_path, f'start_{name}_iter{i}.png'))

    cond_prob_file.close()

    return mode_controllers, np.array(log_info)
