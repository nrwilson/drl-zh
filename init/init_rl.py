from classes.grid import Grid
from classes.grid_mdp import GridMDP
from util.gridworld import run_simulation, RANDOM_POLICY


import numpy as np

# Samples 1M values from a normal distribution with mean pi.
samples = np.random.normal(loc=3.14159265, scale=1.618033, size=(1_000_000,))


def estimate_mean(samples, alpha=0.0001) -> float:
    """Approximates the expected value using exponential smoothing."""
    value = samples[0]
    for sample in samples[1:]:
        # Update the value using the exponential smoothing formula.
        value = value + alpha * (sample-value)
    return value


def init_rl():

    WORLD_GRID = Grid(
        [
            "EEEEE",
            "EWTNG",
            "SEEEW",
        ]
    )

    MDP = GridMDP(WORLD_GRID, gamma=0.9)

    # run_simulation(MDP, RANDOM_POLICY)

    # Samples 1M values from a normal distribution with mean pi.
    samples = np.random.normal(loc=3.14159265, scale=1.618033, size=(10_000_000,))
    # This should print a value very close to 3.14 :)
    print(f"Estimated expected value: {estimate_mean(samples):.2f}")
    return MDP
