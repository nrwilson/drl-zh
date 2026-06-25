from classes.grid_env import GridEnv
from classes.q_table import QTable
from methods.greedy_policy import epsilon_generator, epsilon_greedy_policy, greedy_policy



def q_learning(env: GridEnv, num_episodes, alpha=0.02, max_t=10):
    """Runs Q Learning."""
    # Initialize the QTable, and the epsilon generator.
    states = env.mdp.all_states
    actions = env.mdp.all_actions
    Q = QTable(states, actions)
    epsilon = epsilon_generator(eps_start=1.0, eps_decay=0.999999, eps_min=0.05)
    # Run for the maximum number of episodes passed as input.
    for i_episode in range(1, num_episodes + 1):
        t = 0
        state = env.reset()
        while True:
            # Select an action with an epsilon greedy policy using Q.
            policy = epsilon_greedy_policy(Q, next(epsilon))
            action = policy(state)
            # Interact with the environment
            next_state, reward, done = env.step(action)
            # Store the current Q(s,a) value.
            cur_value = Q[state, action]
            # Determine the next_action using maxQ.
            next_action = Q.best_action(next_state)
            # Compute the TD target.
            td_target = reward + env.mdp.gamma * Q[next_state, next_action]
            # Compute the TD error.
            td_error = td_target - cur_value
            # Update Q with the temporal-difference update rule.
            Q[state, action] = cur_value + alpha * td_error

            # Update the state for the next cycle, and check for episode completion.
            state = next_state
            t = t + 1
            if done or t >= max_t:
                break
        # Monitor and debugging messages.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
    # TODO: Return the optimal policy as the greedy policy on Q.
    policy = greedy_policy(Q)
    return policy, Q