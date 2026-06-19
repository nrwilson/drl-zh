from tests.test_grid import test_grid
from tests.test_grid_mdp import test_grid_mdp
from init import init_simulation
from tests.test_q_table import test_q_table
from tests.test_value_iteration import test_value_iteration


# init_simulation
grid = test_grid()
mdp = test_grid_mdp(grid)
test_q_table()
test_value_iteration(mdp)
