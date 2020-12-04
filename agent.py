"""
References used in agent.py and misc.py:
https://www.nature.com/articles/nature14236
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from misc import ReplayBuffer
from misc import Ann


class Dqn:
    def __init__(self, num_states=8, num_actions=4, buffer_capacity=100000, batch_size=100, network_struct=(50, 50),
                 gamma=0.99, alpha=0.0005, c=10, seed=123456789):
        self.seed = random.seed(seed)
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.D = ReplayBuffer(buffer_capacity, batch_size, seed=seed)
        self.network = Ann(num_states, num_actions, network_struct=network_struct, seed=seed)
        self.target_network = Ann(num_states, num_actions, network_struct=network_struct, seed=seed)
        self.optimizer = optim.Adam(self.network.parameters(), lr=alpha)
        self.step = 0
        self.C = c

    def take_action(self, state, epsilon):
        """
        :param state:
        :param epsilon:
        :return:

        This function chooses the best action using an epsilon-greedy approach.
        """
        states = torch.FloatTensor(list(state))

        self.network.eval()
        with torch.no_grad():
            values_for_actions = self.network(states)
        self.network.train()

        # "with probability epsilon select a random action" according to Algorithm 1 in Mnih et al. (2015)
        if random.random() < epsilon:
            action = random.randrange(self.num_actions)
        # "otherwise select action = argmax(Q)" according to Algorithm 1 in Mnih et al. (2015)
        else:
            action = np.argmax(values_for_actions.data.numpy())

        return action

    def learn(self, state, action, state_next, reward, done):
        """
        :param state:
        :param action:
        :param state_next:
        :param reward:
        :param done:
        :return:

        This functions does the DQN learning according to Algorithm 1 in Mnih et al. (2015) which can found here:
        https://www.nature.com/articles/nature14236
        """
        # "store transition in D" according to Algorithm 1 in Mnih et al. (2015)
        self.D.push(state, action, state_next, reward, done)

        if self.D.get_length() > self.batch_size:
            # "sample minibatch of transitions from D" according to Algorithm 1 in Mnih et al. (2015)
            transitions = self.D.sample()
            transitions = list(map(list, zip(*transitions)))
            states = torch.FloatTensor(transitions[0])
            actions = torch.LongTensor([transitions[1]]).t()
            state_nexts = torch.FloatTensor(transitions[2])
            rewards = torch.FloatTensor([transitions[3]]).t()
            dones = torch.FloatTensor([transitions[4]]).t()

            # calculating the error, loss, according to Algorithm 1 in Mnih et al. (2015)
            Q_hat = self.target_network(state_nexts).detach().max(1)[0].unsqueeze(1)
            y = rewards + (self.gamma * Q_hat * (1 - dones))
            Q = self.network(states).gather(1, actions)
            loss = F.mse_loss(y, Q)

            # "perform gradient descent on mse error" according to Algorithm 1 in Mnih et al. (2015)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # "every C steps reset Q_hat = Q" according to Algorithm 1 in Mnih et al. (2015)
            if (self.step % self.C) == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            self.step += 1
