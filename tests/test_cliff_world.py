from util.gridworld import plot_grid
from tests.test_value_iteration import value_iteration
from classes.state import State
from classes.grid import Grid
from classes.grid_mdp import GridMDP
from classes.grid_env import GridEnv


def test_cliff_world():
    # N has -10 return, G has +10 return.

    CLIFF_WORLD_GRID = Grid(
        [
            "EEEEE",
            "EWEEE",
            "EWTWG",
            "SEEEE",
            "NNNNN",
        ]
    )

    gamma = 0.9
    noise = 0.2
    n_iterations = 100

    # Define a Markov Decision Process.
    mdp = GridMDP(CLIFF_WORLD_GRID, start=State(0, 1), gamma=gamma)

    # Train a q table to estimate values based on states and actions.
    q_table = value_iteration(mdp, noise=noise, n_iterations=n_iterations)

    # Navigate the env using that qtable.
    grid_env = GridEnv(mdp)
    while not grid_env.terminated:
        result = grid_env.smart_step(q_table, noise)
        # result = grid_env.random_step(q_table, noise)
        print(result)

    plot_grid(CLIFF_WORLD_GRID, agent_pos=(0, 1))
