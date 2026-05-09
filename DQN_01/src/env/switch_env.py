import numpy as np
import random
from src.actions import Action
from src.rewards import get_reward
from src.states import normalize_state

class SwitchEnvironment:
    def __init__(self):
        self.max_steps = 200
        self.current_step = 0
        self.state = None

    def reset(self):
        self.current_step = 0
        raw_state = {
            "mac_table_entries": random.uniform(0.0, 1.0),
            "flood_pressure":    random.uniform(0.0, 1.0),
            "port_traffic":      random.uniform(0.0, 1.0),
            "entry_age":         random.uniform(0.0, 1.0),
            "new_mac_rate":      random.uniform(0.0, 1.0),
        }
        self.state = normalize_state(raw_state)
        return self.state

    def step(self, action):
        self.current_step += 1

        reward = get_reward(action, self.state)

        next_raw_state = self._simulate_next_state(action)
        next_state = normalize_state(next_raw_state)
        self.state = next_state

        done = self._is_done()

        return next_state, reward, done

    def _simulate_next_state(self, action):
        mac     = self.state[0]
        flood   = self.state[1]
        traffic = self.state[2]
        age     = self.state[3]
        new_mac = self.state[4]

        if action == Action.LEARN_MAC:
            mac     = min(1.0, mac + new_mac * 0.1)
            flood   = max(0.0, flood - 0.1)

        elif action == Action.EVICT_ENTRY:
            mac     = max(0.0, mac - 0.2)
            age     = max(0.0, age - 0.3)

        elif action == Action.FLOOD:
            flood   = min(1.0, flood + 0.2)
            traffic = min(1.0, traffic + 0.1)

        elif action == Action.BLOCK_PORT:
            traffic = max(0.0, traffic - 0.3)
            new_mac = max(0.0, new_mac - 0.2)

        elif action == Action.UNBLOCK_PORT:
            traffic = min(1.0, traffic + 0.2)
            new_mac = min(1.0, new_mac + 0.1)

        elif action == Action.INCREASE_AGING_TIMER:
            age     = min(1.0, age + 0.2)
            mac     = min(1.0, mac + 0.05)

        elif action == Action.DECREASE_AGING_TIMER:
            age     = max(0.0, age - 0.2)
            mac     = max(0.0, mac - 0.05)

        # Add small random noise to simulate real network unpredictability
        noise = lambda: random.uniform(-0.02, 0.02)

        return {
            "mac_table_entries": np.clip(mac     + noise(), 0.0, 1.0),
            "flood_pressure":    np.clip(flood   + noise(), 0.0, 1.0),
            "port_traffic":      np.clip(traffic + noise(), 0.0, 1.0),
            "entry_age":         np.clip(age     + noise(), 0.0, 1.0),
            "new_mac_rate":      np.clip(new_mac + noise(), 0.0, 1.0),
        }

    def _is_done(self):
        mac   = self.state[0]
        flood = self.state[1]

        if mac >= 0.99:        # Table completely full — overflow
            return True
        if flood >= 0.95:      # Flooding completely out of control
            return True
        if self.current_step >= self.max_steps:
            return True

        return False