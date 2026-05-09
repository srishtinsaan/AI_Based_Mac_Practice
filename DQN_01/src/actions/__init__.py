from enum import IntEnum

class Action(IntEnum):
    LEARN_MAC = 0
    EVICT_ENTRY = 1
    FLOOD = 2
    BLOCK_PORT = 3
    UNBLOCK_PORT = 4
    INCREASE_AGING_TIMER = 5
    DECREASE_AGING_TIMER = 6

NUM_ACTIONS = len(Action)
ACTION_NAMES = [a.name for a in Action]