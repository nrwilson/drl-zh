from tests import test_grid_env
from tests.test_grid import test_grid
from tests.test_grid_mdp import test_grid_mdp
from init import init_simulation
from tests.test_q_table import test_q_table
from tests.test_value_iteration import test_value_iteration
from tests.test_grid_env import test_grid_env
from tests.test_cliff_world import test_cliff_world
from typing import Callable

from init.init_rl import init_rl
from methods.policies import Policy, Episode
from classes.state import State
from classes.action import Action
from tests.test_episode import test_episode
from tests.test_greedy_policy import test_greedy_policy, test_epsilon_greedy_policy, test_epsilon_generator
from tests.test_monte_carlo import test_monte_carlo


# Notebook 1 - Grid
grid = test_grid()
mdp = test_grid_mdp(grid)
test_q_table()
test_value_iteration(mdp)
# test_grid_env(grid)
# test_cliff_world()

# Notebook 2 - RL
mdp2 = init_rl()
assert Policy == Callable[[State], Action]
assert Episode == list[tuple[State, Action, float]]
test_episode(mdp2)
test_greedy_policy()
test_epsilon_greedy_policy()
test_epsilon_generator()
test_monte_carlo(mdp2)