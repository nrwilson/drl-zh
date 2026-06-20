from tests import test_grid_env
from tests.test_grid import test_grid
from tests.test_grid_mdp import test_grid_mdp
from init import init_simulation
from tests.test_q_table import test_q_table
from tests.test_value_iteration import test_value_iteration
from tests.test_grid_env import test_grid_env
from tests.test_cliff_world import test_cliff_world


# init_simulation
grid = test_grid()
mdp = test_grid_mdp(grid)
test_q_table()
test_value_iteration(mdp)
test_grid_env(grid)
test_cliff_world()

