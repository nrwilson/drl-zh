from classes.action import Action
from classes.state import State
from classes.grid_env import GridEnv

from typing import Callable, TypeAlias


# For simplicity, we define a policy as a function that returns an action given a state.
# Define the 'Policy' typealias.
Policy: TypeAlias = Callable[[State], Action]

# An episode instead is a list of tuples (state, action, reward).
# Define the 'Episode' typealias.
Episode: TypeAlias = list[tuple[State, Action, float]]


def generate_episode(env: GridEnv, policy: Policy, noise: float=0.0, max_t=10) -> Episode:
    """Generates an Monte Carlo episode in the Grid World environment GridEnv.

    `max_t` caps episode length so a degenerate policy (e.g., one that bounces between
    cells) cannot spin forever; 10 steps is plenty for our 3x5 grid.
    """
    t = 0
    episode = []
    state = env.reset()
    while t < max_t:
        # Select an action via the policy
        action: Action = policy(state)    
        # Get next_state, reward, done from the environment.
        next_state, reward, done = env.step(action, noise=noise)
        # Record the step in the episode list.
        new_episode = (state, action, reward)
        episode.append(new_episode)
        # Update state, time, and check for completion.
        state = next_state
        if done:
            break
        t += 1
        
    return episode


