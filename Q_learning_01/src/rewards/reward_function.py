def get_reward(outcome):
    """
    Calculates the reward based on the network outcome[cite: 112].
    
    Positive rewards reinforce good behaviors (efficiency, stability) [cite: 96-101].
    Negative rewards penalize errors (congestion, security risks) [cite: 102-111].
    """
    rewards_map = {
        # Positive Outcomes
        "correct_delivery": 1.0,         # Successful MAC lookup [cite: 97]
        "flooding_avoided": 0.8,         # Efficient forwarding [cite: 99]
        "preventive_removal": 0.3,       # Table maintenance [cite: 101]
        
        # Negative Outcomes
        "necessary_flooding": -0.1,      # Fallback for unknown MAC [cite: 103]
        "unnecessary_flooding": -1.0,    # Critical efficiency error [cite: 105]
        "wrong_port_stale": -0.5,        # Accuracy error [cite: 107]
        "early_removal": -0.3,           # Wasteful eviction [cite: 109]
        "table_overflow": -0.7           # Capacity management failure [cite: 111]
    }
    
    # Return 0.0 if the outcome string doesn't match (safety default)
    return rewards_map.get(outcome, 0.0)