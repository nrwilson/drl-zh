from tests.test_grid import test_grid
from tests.test_grid_mdp import test_grid_mdp
from init import init_simulation

init_simulation
grid = test_grid()
test_grid_mdp(grid)