import numpy as np
import random

class QAgent:
    def __init__(self, states=3, actions=6, alpha=0.1, gamma=0.9, 
                 epsilon_start=1.0, epsilon_min=0.01, decay_rate=0.995):
        self.q_table = np.zeros((states, actions)) # [cite: 15, 165-185]
        self.alpha = alpha     # Learning rate [cite: 31, 218]
        self.gamma = gamma     # Discount factor [cite: 30, 219]
        self.epsilon = epsilon_start 
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate

    def choose_action(self, state):
        # Epsilon-Greedy Algorithm [cite: 20, 213-215]
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 5)  # Exploration [cite: 21, 24]
        return np.argmax(self.q_table[state]) # Exploitation [cite: 22, 28-29]

    def decay_epsilon(self):
        """Reduces epsilon over time to shift from explore to exploit."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)

    def update(self, s, a, r, s_next):
        # Bellman Equation: Q(s,a) = Q(s,a) + α[R + γ * maxQ(s',a') - Q(s,a)] [cite: 19, 220]
        old_q = self.q_table[s, a]
        next_max = np.max(self.q_table[s_next])
        self.q_table[s, a] = old_q + self.alpha * (r + self.gamma * next_max - old_q)