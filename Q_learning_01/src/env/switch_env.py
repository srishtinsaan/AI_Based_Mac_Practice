import pandas as pd
from src.states.states import get_state

class SwitchEnv:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0
        row = self.data.iloc[self.current_idx]
        return get_state(row['mac_fill'], row['flood_pressure'], row['entry_age'])

    def step(self, action):
        # Determine reward based on State + Action mapping [cite: 368-379]
        row = self.data.iloc[self.current_idx]
        current_state = get_state(row['mac_fill'], row['flood_pressure'], row['entry_age'])
        
        # Reward Logic [cite: 95-112]
        reward = 0
        if current_state == 0 and action == 0: reward = 1.0  # S1 + LEARN [cite: 372]
        elif current_state == 2 and action == 4: reward = 0.9 # S3 + BLOCK [cite: 378]
        elif action == 2: reward = -1.0                       # Bad Flood [cite: 105]

        # Advance in data table
        self.current_idx = (self.current_idx + 1) % len(self.data)
        next_row = self.data.iloc[self.current_idx]
        next_state = get_state(next_row['mac_fill'], next_row['flood_pressure'], next_row['entry_age'])
        
        return next_state, reward, False