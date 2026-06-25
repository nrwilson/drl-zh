from classes.grid_env import GridEnv
from classes.q_table import QTable
from methods.greedy_policy import epsilon_generator, epsilon_greedy_policy, greedy_policy


def sarsa(env: GridEnv, num_episodes, alpha=0.02, max_t=10):
    """Runs SARSA (on-policy TD)."""
    # Initialize the QTable and the epsilon generator (same as Q-Learning).
    Q = QTable(env.mdp.all_states, env.mdp.all_actions)
    # Setting eps_decay to 0.99999 helps optimize- otherwise it explores too long and doesn't find the exploit.
    epsilon = epsilon_generator(eps_start=1.0, eps_decay=0.999999, eps_min=0.05)
    for i_episode in range(1, num_episodes + 1):
        t = 0
        state = env.reset()
        # SARSA needs the next action *before* doing the TD update, so pick the first action up
        # front and then advance state and action together with the environment.
        # Build an epsilon-greedy policy and sample the first action.
        policy = epsilon_greedy_policy(Q, next(epsilon))
        action = policy(state)
        while True:
            # Interact with the environment.
            next_state, reward, done = env.step(action)
            # Refresh the epsilon-greedy policy and sample the *next* action with it.
            #       (This is what makes SARSA on-policy: target uses the behavior policy.)
            policy = epsilon_greedy_policy(Q, next(epsilon))
            next_action = policy(next_state)
            # Store the current Q(s,a) value.
            current_value = Q[state, action]
            # Compute the TD target using Q(next_state, next_action) -- NOT max Q.
            td_target = reward + env.mdp.gamma * Q[next_state, next_action]
            # TD error and Q update (same structure as Q-Learning).
            td_error = td_target - current_value
            Q[state, action] = current_value + alpha * td_error
            # Advance state AND action in lockstep (key difference vs. Q-Learning).
            state, action = next_state, next_action
            t = t + 1
            if done or t >= max_t:
                break
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
    # TODO: Return the greedy policy on Q.
    policy = greedy_policy(Q)
    return policy, Q
