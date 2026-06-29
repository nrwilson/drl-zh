from collections import deque
from typing import NamedTuple
import numpy as np
import torch
from util.gymnastics import DEVICE


class Experience(NamedTuple):
    """An Experience stored in the replay buffer."""

    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """The replay buffer for DQN."""

    def __init__(self, buffer_size=int(1e4)):
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        stored_action = np.atleast_1d(action)
        e = Experience(state, stored_action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size: int = 32):
        all_indices = np.arange(len(self.memory))
        selection = np.random.choice(all_indices, size=batch_size)
        return self.unpack(selection)

    def unpack(self, selection):
        experiences = [e for i in selection if (e := self.memory[i]) is not None]
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.stack(states)).float().to(DEVICE)
        actions = torch.from_numpy(np.stack(actions)).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(DEVICE)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack(dones, dtype=np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


def test_replay_buffer():
    # Replay buffer minimal test.
    test_buffer = ReplayBuffer()


    def fake_state():
        return np.random.rand(4, 5)


    test_buffer.add(fake_state(), [10], 1.0, fake_state(), False)
    test_buffer.add(fake_state(), [11], 2.0, fake_state(), False)
    test_buffer.add(fake_state(), 12, 3.0, fake_state(), False)
    test_buffer.add(fake_state(), 13, 4.0, fake_state(), False)
    test_buffer.add(fake_state(), [14], 5.0, fake_state(), True)

    batch_size = 3
    state_shape = (batch_size, 4, 5)

    t_states, t_actions, t_rewards, t_next_states, t_dones = test_buffer.sample(batch_size)

    assert t_states.shape == state_shape
    assert t_actions.shape == (batch_size, 1)
    assert t_rewards.shape == (batch_size, 1)
    assert t_next_states.shape == state_shape
    assert t_dones.shape == (batch_size, 1)