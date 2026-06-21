from classes.action import Action
from classes.state import State
from classes.q_table import QTable
from methods.greedy_policy import greedy_policy, epsilon_greedy_policy, epsilon_generator

def test_greedy_policy():

    # Test the implementation
    state_0 = State(0, 0)
    state_1 = State(1, 0)
    test_qtable = QTable([state_0, state_1], list(Action))

    test_qtable[state_0, Action.DOWN] = 0.5
    test_qtable[state_0, Action.LEFT] = 1.5
    test_qtable[state_0, Action.RIGHT] = 0.8
    test_qtable[state_1, Action.UP] = 0.1

    test_greedy_policy = greedy_policy(test_qtable)

    assert test_greedy_policy(state_0) == Action.LEFT
    assert test_greedy_policy(state_1) == Action.UP

def test_epsilon_greedy_policy():
    # Approximate tests... hopefully good enough to find big bugs :)
    state_0 = State(0, 0)
    state_1 = State(1, 0)
    test_qtable = QTable([state_0, state_1], list(Action))

    test_qtable[state_0, Action.DOWN] = 0.5
    test_qtable[state_0, Action.LEFT] = 1.5
    test_qtable[state_0, Action.RIGHT] = 0.8
    test_qtable[state_1, Action.UP] = 0.1


    def probe_actions(policy, state) -> list[Action]:
        return dict.fromkeys([policy(state) for _ in range(5_000)])


    test_egreedy_policy = epsilon_greedy_policy(test_qtable, epsilon=1.0)
    assert len(probe_actions(test_egreedy_policy, state_0)) == 4
    assert len(probe_actions(test_egreedy_policy, state_1)) == 4

    test_egreedy_policy = epsilon_greedy_policy(test_qtable, epsilon=0.0)
    assert len(probe_actions(test_egreedy_policy, state_0)) == 1
    assert len(probe_actions(test_egreedy_policy, state_1)) == 1


def test_epsilon_generator():
    # Test the implementation!
    eps = epsilon_generator(1.0, 0.5, 0.1)

    assert next(eps) == 1.0
    assert next(eps) == 0.5
    assert next(eps) == 0.25
    assert next(eps) == 0.125
    assert next(eps) == 0.1
    assert next(eps) == 0.1