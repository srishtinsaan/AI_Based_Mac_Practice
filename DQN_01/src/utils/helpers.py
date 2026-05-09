import numpy as np
import torch
import random
import os
from collections import deque
from src.utils.config import config

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=config["buffer_size"])

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=None):
        batch_size = batch_size or config["batch_size"]
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_model(agent, path="models/dqn_model.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "prediction_network": agent.prediction_network.state_dict(),
        "target_network":     agent.target_network.state_dict(),
        "epsilon":            agent.epsilon,
        "step_count":         agent.step_count,
    }, path)
    print(f"Model saved to {path}")


def load_model(agent, path="models/dqn_model.pth"):
    checkpoint = torch.load(path)
    agent.prediction_network.load_state_dict(checkpoint["prediction_network"])
    agent.target_network.load_state_dict(checkpoint["target_network"])
    agent.epsilon    = checkpoint["epsilon"]
    agent.step_count = checkpoint["step_count"]
    print(f"Model loaded from {path}")