# Test the implementation!
from numpy.testing import assert_almost_equal
from classes.grid_mdp import GridMDP
from classes.state import State
from classes.action import Action
from classes.grid import Grid
from init.init_simulation import init_random

def test_grid_mdp(grid: Grid):

    init_random()
    NOISE = 0.2
    GAMMA = 0.9

    GRID_MDP = GridMDP(grid, gamma=GAMMA)

    # Test all states
    all_states = GRID_MDP.all_states
    assert len(all_states) == 4 * 3
    assert GRID_MDP.all_actions == list(Action)

    # Test terminal and reachable states
    for s in all_states:
        if s == State(3, 2) or s == State(3, 1):
            assert GRID_MDP.is_terminal(s)
            assert GRID_MDP.is_reachable(s)
        elif s == State(1, 1):
            assert not GRID_MDP.is_reachable(s)
        else:
            assert not GRID_MDP.is_terminal(s)
            assert GRID_MDP.is_reachable(s)

    # Test rewards. These are the *environment* rewards: gamma does not appear here — it
    # is applied only inside the Bellman backup, see `value_iteration` further below.
    assert GRID_MDP.reward(State(0, 2), Action.DOWN, State(0, 1)) == 0.0
    assert GRID_MDP.reward(State(2, 1), Action.RIGHT, State(3, 1)) == -1.0
    assert GRID_MDP.reward(State(2, 2), Action.RIGHT, State(3, 2)) == 1.0

    # Test transitions
    p = GRID_MDP.get_transition_probabilities(State(1, 2), Action.RIGHT, noise=0.2)
    assert_almost_equal(p[State(1, 2)], 0.2)
    assert_almost_equal(p[State(2, 2)], 0.8)

    p = GRID_MDP.get_transition_probabilities(State(2, 0), Action.UP, noise=0.0)
    assert_almost_equal(p[State(2, 1)], 1.0)
    assert_almost_equal(p[State(2, 0)], 0.0)
    assert_almost_equal(p[State(1, 0)], 0.0)
    assert_almost_equal(p[State(3, 0)], 0.0)
    assert State(1, 1) not in p

    p = GRID_MDP.get_transition_probabilities(State(1, 0), Action.UP, noise=0.0)  # Hitting wall, for sure stays.
    assert_almost_equal(p[State(1, 0)], 1.0)
    assert_almost_equal(p[State(2, 0)], 0.0)
    assert_almost_equal(p[State(0, 0)], 0.0)

    p = GRID_MDP.get_transition_probabilities(State(2, 0), Action.UP, noise=0.2)
    assert_almost_equal(p[State(2, 1)], 0.8)
    assert_almost_equal(p[State(1, 0)], 0.1)
    assert_almost_equal(p[State(3, 0)], 0.1)
    assert_almost_equal(p[State(2, 0)], 0.0)

    p = GRID_MDP.get_transition_probabilities(State(0, 0), Action.DOWN, noise=0.2)
    assert_almost_equal(p[State(0, 0)], 0.9)
    assert_almost_equal(p[State(1, 0)], 0.1)
    assert_almost_equal(p[State(0, 1)], 0.0)

    p = GRID_MDP.get_transition_probabilities(State(0, 0), Action.UP, noise=0.2)
    assert_almost_equal(p[State(0, 0)], 0.1)
    assert_almost_equal(p[State(1, 0)], 0.1)
    assert_almost_equal(p[State(0, 1)], 0.8)

    p = GRID_MDP.get_transition_probabilities(State(2, 1), Action.LEFT, noise=0.2)
    assert_almost_equal(p[State(2, 1)], 0.8)
    assert_almost_equal(p[State(2, 2)], 0.1)
    assert_almost_equal(p[State(2, 0)], 0.1)

    return GRID_MDP
