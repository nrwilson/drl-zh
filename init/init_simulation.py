# For reproducibility, all the examples have fixed random seed.
import random
import numpy as np


def init_random(random_seed=10):
    """Initializes the random generators used in the code to a predetermined random seed."""
    random.seed(random_seed)
    np.random.seed(random_seed)


# Import some predefined constants and utility functions for visualization!
from util.gridworld import RANDOM_POLICY, GRID_WORLD_MDP, run_simulation

init_random()
run_simulation(GRID_WORLD_MDP, RANDOM_POLICY)