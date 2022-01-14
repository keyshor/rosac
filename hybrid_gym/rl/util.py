def get_rollout(env, policy, render, max_timesteps=10000):
    '''
    Compute a single rollout.

    env: Environment
    policy: Policy
    render: bool
    return: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]
             ((state, action, reward, next_state) tuples)
    '''
    # Step 1: Initialization
    state = env.reset()
    done = False

    # Step 2: Compute rollout
    sarss = []
    steps = 0
    while (not done) and (steps < max_timesteps):
        # Step 2a: Render environment
        if render:
            env.render()

        # Step 2b: Action
        action = policy.get_action(state)

        # Step 2c: Transition environment
        next_state, reward, done, _ = env.step(action)

        # Step 2d: Rollout (s, a, r)
        sarss.append((state, action, reward, next_state))

        # Step 2e: Update state
        state = next_state
        steps += 1

    # Step 3: Render final state
    if render:
        env.render()

    return sarss


def discounted_reward(sarss, gamma):
    sarss_rev = sarss.copy()
    sarss_rev.reverse()
    reward = 0.0
    for _, _, r, _ in sarss_rev:
        reward = r + gamma*reward
    return reward


def test_policy(env, policy, n_rollouts, gamma=1, use_cum_reward=False, max_timesteps=10000,
                get_steps=False):
    '''
    Estimate the cumulative reward of the policy.

    env: Environment
    policy: Policy
    n_rollouts: int
    return: float
    '''
    cum_reward = 0.0
    num_steps = 0
    for _ in range(n_rollouts):
        sarss = get_rollout(env, policy, False, max_timesteps=max_timesteps)
        num_steps += len(sarss)
        if use_cum_reward:
            cum_reward += env.cum_reward(sarss)
        else:
            cum_reward += discounted_reward(sarss, gamma)
    if get_steps:
        return cum_reward / n_rollouts, num_steps
    return cum_reward / n_rollouts


def get_reach_prob(env, policy, n_rollouts, max_timesteps=10000):
    '''
    Estimate the probability of reaching the goal.
    Works only for 0-1 rewards.

    env: Environment
    policy: Policy
    n_rollouts: int
    return: float
    '''
    succesful_trials = 0
    for _ in range(n_rollouts):
        sarss = get_rollout(env, policy, False, max_timesteps=max_timesteps)
        if discounted_reward(sarss, 1) > 0:
            succesful_trials += 1
    return succesful_trials / n_rollouts


# print reward and reaching probability
def print_performance(env, policy, gamma, n_rollouts=100, max_timesteps=10000):
    reward = test_policy(env, policy, n_rollouts, gamma=gamma, max_timesteps=max_timesteps)
    reach_prob = get_reach_prob(env, policy, n_rollouts, max_timesteps=max_timesteps)
    print('\nEstimated Reward: {}'.format(reward))
    print('Estimated Reaching Probability: {}'.format(reach_prob))
    return reward, reach_prob
