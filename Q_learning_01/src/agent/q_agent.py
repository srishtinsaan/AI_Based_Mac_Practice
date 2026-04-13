import numpy as np
import random

class QAgent:
    def __init__(self, states, actions=6, alpha=0.1, gamma=0.9):
        self.q_table = np.zeros((states, actions))
        self.alpha, self.gamma = alpha, gamma
        self.epsilon = 1.0

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 5)
        return np.argmax(self.q_table[state])

    def update(self, s, a, r, s_next):
        old_val = self.q_table[s, a]
        next_max = np.max(self.q_table[s_next])
        self.q_table[s, a] = old_val + self.alpha * (r + self.gamma * next_max - old_val)

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)