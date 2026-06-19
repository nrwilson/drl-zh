# Test our implementation!
from classes.state import State
from classes.action import Action
from classes.q_table import QTable
from init.init_simulation import init_random


def test_q_table():
    init_random()

    state_0 = State(0, 0)
    state_1 = State(10, 10)
    state_2 = State(11, 11)
    state_3 = State(11, 10)
    qtable = QTable([state_0, state_1, state_2, state_3], list(Action))

    # props
    assert qtable.states == [state_0, state_1, state_2, state_3]
    assert qtable.actions == list(Action)
    assert qtable.num_actions == len(list(Action))

    # init get
    assert qtable[(State(10, 10), Action.DOWN)] == 0.0

    # set
    qtable[state_1, Action.DOWN] = 0.5
    qtable[state_1, Action.UP] = 1.5
    qtable[state_2, Action.LEFT] = 2.5
    assert qtable[state_1, Action.DOWN] == 0.5
    assert qtable[state_1, Action.UP] == 1.5
    assert qtable[state_2, Action.LEFT] == 2.5
    assert qtable[state_2, Action.RIGHT] == 0.0

    # value
    assert qtable.value(state_1) == 1.5
    assert qtable.value(state_2) == 2.5
    assert qtable.value(state_3) == 0.0

    # best action
    assert qtable.best_action(state_1) == Action.UP
    assert qtable.best_action(state_2) == Action.LEFT
    # let's hope large numbers don't fail us :)
    many_actions = [qtable.best_action(state_0) for _ in range(5_000)]
    assert len(dict.fromkeys(many_actions)) == 4