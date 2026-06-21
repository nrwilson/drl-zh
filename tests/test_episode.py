from classes.grid_env import GridEnv
from classes.action import Action
from classes.state import State
from classes.grid_mdp import GridMDP
from methods.policies import generate_episode


def test_episode(mdp: GridMDP):
    # Test
    test_env = GridEnv(mdp)
    test_policy = lambda _: Action.LEFT
    test_episode = generate_episode(test_env, test_policy, max_t=3)
    assert len(test_episode) == 3
    assert test_episode[0] == (State(0, 0), Action.LEFT, 0.0)
    assert test_episode[1] == (State(0, 0), Action.LEFT, 0.0)
    assert test_episode[2] == (State(0, 0), Action.LEFT, 0.0)

    test_policy = lambda s: Action.UP if s == State(3, 0) else Action.RIGHT
    test_episode = generate_episode(test_env, test_policy)
    assert len(test_episode) == 4
    assert test_episode[0] == (State(0, 0), Action.RIGHT, 0.0)
    assert test_episode[1] == (State(1, 0), Action.RIGHT, 0.0)
    assert test_episode[2] == (State(2, 0), Action.RIGHT, 0.0)
    # Stepping UP from (3, 0) lands on the NUKE cell (3, 1) — environment reward is -10.0
    # (we no longer scale rewards by gamma inside the env).
    assert test_episode[3] == (State(3, 0), Action.UP, -10.0)