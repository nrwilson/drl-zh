from struct import unpack
from classes.state import State
from classes.action import Action
from classes.q_table import QTable
from classes.grid_env import GridEnv
from methods.policies import Episode, Policy
import math
from methods.greedy_policy import epsilon_generator, epsilon_greedy_policy, greedy_policy
from methods.policies import generate_episode


def monte_carlo(env: GridEnv, num_episodes, alpha=0.02, eps_start=1.0, start_q: QTable = None) -> (Policy, QTable):
    """Given an environment and # training episodes to run, return a trained policy and Q table.
    
    A Monte Carlo algorithm for reinforcement learning."""
    # Initialize the QTable (use start_q if provided, you'll see why later)
    Q = start_q

    # Prepare to generate epsilon creating the generator.
    epsilon = epsilon_generator()

    # Iterate until we reached the maximum number of episodes for learning.
    for i_episode in range(1, num_episodes + 1):
        # Use the epsilon-greedy policy.
        policy = epsilon_greedy_policy(Q, next(epsilon))
        # Generate an episode from trying that policy.
        episode = generate_episode(env, policy)

        # Find out what happened.
        states, actions, rewards = unpack_episode(episode)

        # Compute discount for each time point.
        discounts = get_discounts(rewards, gamma=env.mdp.gamma)

        # Update the q table based on what was observed.
        Q = update_q_table(Q, states, actions, rewards, discounts, alpha)

        # Monitor progress
        if i_episode % 1000 == 0:
            print(f"\rEpisode {i_episode}/{num_episodes}.", end="")

    # Define the optimal policy (i.e., the greedy policy on the computed QTable).
    policy = greedy_policy(Q)
    return policy, Q


def unpack_episode(episode: Episode) -> ([State], [Action], [float]):
    # Unpack the episode in a tuple of (list[states], list[actions], list[rewards]).
    #       Hint: use the zip function!
    states, actions, rewards = zip(*episode)
    return states, actions, rewards

def get_discounts(rewards: [float], gamma: float) -> [float]:
    # Conveniently compute the discounts first. We can do this b/c we can compute all the
    #       expected returns at each timestep (having all the rewards).
    #       The discounts are: [1, gamma, gamma^2, gamma^3, ...] for the length of the episode.
    discounts = []
    for j in range(len(rewards)):
        discount = math.pow(gamma, j)
        discounts.append(discount)
    return discounts


def update_q_table(Q: QTable, states: [State], actions: [Action], rewards: [float], discounts: [float], alpha: float) -> QTable:
    # For each step / transition in the environment, let's update the QTable according to the
    # update rule of Monte Carlo methods defined above.

    # For each state,action in the episode - update the q table based on the reward encountered
    # and the discount based on how far in the future encountered.

    # Get the state, action pair from the episode.
    num_steps = len(states)
    old_Q = Q
    for t, state in enumerate(states):
        action = actions[t]
        # Compute the total return. Recall that:
        #       G_0 = R_1 + gamma * R_2 + gamma^2 * R_3 + ... (R1 is found at index 0)
        #       Hint: sum rewards _from_ `t` onward, while select discounts _until_ `t`. That is
        #       because discounts always start from the beginning even if rewards "shift".
        these_rewards = rewards[t: num_steps-t]
        G_t = 0
        for i, reward in enumerate(these_rewards):
            discount = discounts[i]
            discounted_reward = reward * discount
            G_t += discounted_reward
        Q[state, action] = old_Q[state, action] + alpha * (G_t - old_Q[state, action])
    return Q
