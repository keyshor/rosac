'''
Utility functions for testing/simulation.
'''
from hybrid_gym.model import Mode, Transition, Controller, ModeSelector
from hybrid_gym.hybrid_env import HybridAutomaton
from typing import List, Any, Dict, Union
import gym


def get_rollout(mode: Mode, transitions: List[Transition], controller: Controller,
                state: Any = None, max_timesteps=10000, reset_controller: bool = True,
                render: bool = False):
    step = 0
    if reset_controller:
        controller.reset()
    if state is None:
        state = mode.reset()

    sass = []
    info: Dict[str, Any] = {'safe': True, 'jump': None}

    obs_space = mode.observation_space

    while step <= max_timesteps:
        obs = mode.observe(state)
        action = controller.get_action(obs)
        next_state = mode.step(state, action)
        sass.append((state, action, next_state))
        state = next_state

        # Render
        if render:
            mode.render(state)

        # Check safety
        if not mode.is_safe(state):
            info['safe'] = False
            break

        # Check guards of transitions out of mode
        for t in transitions:
            if t.guard(state):
                info['jump'] = t
        if info['jump'] is not None:
            break

        # Increment step count
        step += 1

    return sass, info


def end_to_end_test(automaton: HybridAutomaton, selector: ModeSelector,
                    controller: Union[Controller, Dict[str, Controller]],
                    time_limits: Dict[str, int], num_rollouts: int = 100,
                    max_jumps: int = 100, print_debug: bool = False,
                    render: bool = False):
    '''
    Measure success of trained controllers w.r.t. a given mode selector.
    Success only when selector signals completion (returns done).

    controller: Can be a single controller (also handles mode detection) OR
                one controller per mode (assumes full observability)

    Returns: float (the probability of success)
    '''
    num_success = 0

    for _ in range(num_rollouts):
        steps = 0
        mname = selector.reset()
        state = automaton.modes[mname].end_to_end_reset()
        if isinstance(controller, Controller):
            controller.reset()
        if render:
            print('\n**** New rollout ****')

        for j in range(max_jumps):

            # pick the current controller
            if isinstance(controller, Controller):
                cur_controller = controller
            else:
                cur_controller = controller[mname]

            if render:
                print('Rollout in mode {}'.format(mname))
            sarss, info = get_rollout(automaton.modes[mname], automaton.transitions[mname],
                                      cur_controller, state, time_limits[mname],
                                      reset_controller=(not isinstance(controller, Controller)),
                                      render=render)
            steps += len(sarss)

            # terminate rollout if unsafe
            if info['jump'] is None or not info['safe']:
                if print_debug:
                    if info['safe']:
                        print('Failed to make progress in mode {} after {} jumps'.format(
                            mname, j))
                    else:
                        print('Unsafe state reached in mode {} after {} jumps'.format(
                            mname, j))
                break

            # select next mode
            mname, done = selector.next_mode(info['jump'], sarss[-1][-1])

            # update start state
            if not done:
                state = info['jump'].jump(mname, sarss[-1][-1])

            # count success
            else:
                num_success += 1
                break

    return num_success / num_rollouts
