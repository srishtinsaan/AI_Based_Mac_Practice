import torch
import torch.nn as nn
import numpy as np
import random
import copy
from src.network import DQNetwork
from src.utils import ReplayBuffer, config

class DQNAgent:
    def __init__(self):
        self.prediction_network = DQNetwork()
        self.target_network = copy.deepcopy(self.prediction_network)
        self.target_network.load_state_dict(self.prediction_network.state_dict())

        self.optimizer = torch.optim.Adam(
            self.prediction_network.parameters(), 
            lr=config["alpha"]
        )
        self.loss_fn = nn.MSELoss()

        self.epsilon = config["epsilon_start"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]

        self.step_count = 0
        self.gamma = config["gamma"]

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, config["num_actions"] - 1)  # Explore
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.prediction_network(state_tensor)
            return torch.argmax(q_values).item()              # Exploit

    def train(self, replay_buffer):
        if len(replay_buffer) < config["batch_size"]:
            return

        batch = replay_buffer.sample(config["batch_size"])
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(states)
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones       = torch.FloatTensor(dones)

        # Predicted Q-values for actions taken
        predicted_q = self.prediction_network(states)
        predicted_q = predicted_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values using target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(predicted_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_count += 1

        # Target network update
        if self.step_count % config["target_update_freq"] == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.prediction_network.state_dict())