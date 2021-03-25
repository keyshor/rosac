'''
Utility functions.
'''
from hybrid_gym.model import Mode, Transition, Controller
from typing import List, Any


def get_rollout(mode: Mode, transitions: List[Transition], controller: Controller,
                state: Any = None, max_timesteps=10000):
    step = 0
    controller.reset()
    if state is None:
        state = mode.reset()

    sass = []

    while step <= max_timesteps:
        obs = mode.observe(state)
        action = controller.get_action(obs)
        next_state = mode.step(state, action)
        sass.append((state, action, next_state))
        state = next_state

        # Check safety
        if not mode.is_safe(state):
            break

        # Check guards of transitions out of mode
        jumped = False
        for t in transitions:
            if t.guard(state):
                jumped = True
        if jumped:
            break

        # Increment step count
        step += 1

    return sass
