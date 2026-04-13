import pandas as pd
from src.states.states import get_state

class SwitchEnv:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.current_idx = 0

    def reset(self):
        self.current_idx = 0
        row = self.data.iloc[self.current_idx]
        return get_state(row['mac_fill'], row['flood_pressure'], row['port_traffic'], 
                         row['vlan_info'], row['entry_age'], row['new_mac_rate'])

    def step(self, action):
        # 1. Identify current situation from the CSV row [cite: 94, 113-118]
        row = self.data.iloc[self.current_idx]
        current_s = get_state(row['mac_fill'], row['flood_pressure'], row['port_traffic'], 
                              row['vlan_info'], row['entry_age'], row['new_mac_rate'])
        
        # 2. Apply your Reward Logic [cite: 95-112, 368-379]
        reward = 0
        if current_s == 0: # S1: Normal
            reward = 1.0 if action == 0 else -0.5 # Favor LEARN_MAC 
        elif current_s == 1: # S2: Moderate
            reward = 0.8 if action == 1 else -0.5 # Favor EVICT_ENTRY [cite: 375]
        elif current_s == 2: # S3: Critical
            reward = 0.9 if action == 4 else -1.0 # Favor BLOCK_PORT [cite: 378]

        # 3. Advance to the next network scenario
        self.current_idx = (self.current_idx + 1) % len(self.data)
        next_row = self.data.iloc[self.current_idx]
        
        # 4. Get the next state index [cite: 14, 119-158]
        next_s = get_state(next_row['mac_fill'], next_row['flood_pressure'], next_row['port_traffic'], 
                           next_row['vlan_info'], next_row['entry_age'], next_row['new_mac_rate'])

        # CRITICAL FIX: You must return these three values!
        return next_s, reward, False