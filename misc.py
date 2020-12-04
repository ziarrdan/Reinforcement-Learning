"""
References used in agent.py and misc.py:
https://www.nature.com/articles/nature14236
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import torch.nn.functional as F
from collections import deque
import random
import torch


class ReplayBuffer:
    def __init__(self, buffer_capacity, batch_size, seed=123456789):
        self.seed = random.seed(seed)
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_capacity)

    def push(self, *args):
        self.buffer.append([*args])

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def get_length(self):
        return len(self.buffer)


class Ann(torch.nn.Module):
    def __init__(self, num_states=8, num_actions=4, network_struct=(50, 50), seed=123456789):
        super(Ann, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.in_layer = torch.nn.Linear(num_states, network_struct[0])
        self.mid_layer = torch.nn.Linear(network_struct[0], network_struct[1])
        self.out_layer = torch.nn.Linear(network_struct[1], num_actions)

    def forward(self, input):
        x = F.relu(self.in_layer(input))
        x = F.relu(self.mid_layer(x))
        x = self.out_layer(x)

        return x