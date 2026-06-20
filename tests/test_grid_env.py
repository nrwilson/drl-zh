from classes.action import Action
from classes.state import State
from classes.grid_mdp import GridMDP
from classes.grid_env import GridEnv
from init.init_simulation import init_random


def test_grid_env(grid):
    # Test your implementation!
    init_random()

    GRID_ENV = GridEnv(GridMDP(grid))

    assert not GRID_ENV.terminated

    assert GRID_ENV.step(Action.UP) == (State(0, 1), 0.0, False)
    assert GRID_ENV.state == State(0, 1)

    assert GRID_ENV.step(Action.UP) == (State(0, 2), 0.0, False)
    assert GRID_ENV.state == State(0, 2)

    GRID_ENV.terminated = True
    assert GRID_ENV.reset() == State()
    assert GRID_ENV.state == State()
    assert GRID_ENV.terminated == False

    GRID_ENV.step(Action.UP)
    GRID_ENV.step(Action.UP)
    GRID_ENV.step(Action.RIGHT)

    assert GRID_ENV.step(Action.RIGHT) == (State(2, 2), 0.0, False)
    assert GRID_ENV.step(Action.RIGHT) == (State(3, 2), 1.0, True)
    assert GRID_ENV.terminated