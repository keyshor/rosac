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
    sariss = []
    steps = 0
    while (not done) and (steps < max_timesteps):
        # Step 2a: Render environment
        if render:
            env.render()

        # Step 2b: Action
        action = policy.get_action(state)

        # Step 2c: Transition environment
        next_state, reward, done, info = env.step(action)

        # Step 2d: Rollout (s, a, r)
        sariss.append((state, action, reward, info, next_state))

        # Step 2e: Update state
        state = next_state
        steps += 1

    # Step 3: Render final state
    if render:
        env.render()

    return sariss


def discounted_reward(sariss, gamma, reward_fn=None):
    sariss_rev = sariss.copy()
    sariss_rev.reverse()
    reward = 0.0
    for s, a, r, info, ns in sariss_rev:
        if reward_fn is not None:
            r = reward_fn.obs_reward(s, a, ns, r, info)
        reward = r + gamma*reward
    return reward


def test_policy(env, policy, n_rollouts, gamma=1, max_timesteps=10000,
                get_steps=False, reward_fn=None):
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
        sariss = get_rollout(env, policy, False, max_timesteps=max_timesteps)
        num_steps += len(sariss)
        cum_reward += discounted_reward(sariss, gamma, reward_fn)
    if get_steps:
        return cum_reward / n_rollouts, num_steps
    return cum_reward / n_rollouts
