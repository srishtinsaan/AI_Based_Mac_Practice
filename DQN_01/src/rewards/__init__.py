import numpy as np
from src.actions import Action

def get_reward(action, state):
    mac     = state[0]
    flood   = state[1]
    traffic = state[2]
    age     = state[3]
    new_mac = state[4]

    # Correct MAC lookup and delivery
    if action == Action.LEARN_MAC and flood < 0.3 and mac < 0.85:
        return +1.0

    # Flooding avoided
    if action == Action.EVICT_ENTRY and mac > 0.85 and age > 0.7:
        return +0.8

    # Preventive entry removal before overflow
    if action == Action.EVICT_ENTRY and mac > 0.6 and mac <= 0.85:
        return +0.3

    # Necessary flooding (unknown MAC)
    if action == Action.FLOOD and flood < 0.3:
        return -0.1

    # Unnecessary flooding
    if action == Action.FLOOD and flood >= 0.3:
        return -1.0

    # Wrong port delivery due to stale entry
    if action == Action.LEARN_MAC and age > 0.8:
        return -0.5

    # Early removal of active entry
    if action == Action.EVICT_ENTRY and age < 0.3:
        return -0.3

    # Table overflow
    if mac >= 0.95:
        return -0.7

    return 0.0