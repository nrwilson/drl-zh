from init.init_simulation import init_random
from dqn.q_network import QNetwork
import numpy as np
import torch


def test_q_network():
    # Test for neural network!
    init_random()

    test_net = QNetwork(action_size=6)

    fake_img = np.random.randn(2, 84, 84)
    fake_tensor = torch.from_numpy(fake_img).float()
    result = test_net.forward(fake_tensor).detach()

    expected_result = torch.tensor([[-0.2041, 0.0406, -0.0483, 0.0051, 0.0216, 0.0395]])
    assert torch.allclose(result, expected_result, atol=0.0001)