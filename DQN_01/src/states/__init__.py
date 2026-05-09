import numpy as np

STATE_SIZE = 5

STATE_FEATURES = [
    "mac_table_entries",
    "flood_pressure",
    "port_traffic",
    "entry_age",
    "new_mac_rate"
]

def normalize_state(raw_state: dict) -> np.ndarray:
    return np.array([
        np.clip(raw_state["mac_table_entries"], 0.0, 1.0),
        np.clip(raw_state["flood_pressure"],    0.0, 1.0),
        np.clip(raw_state["port_traffic"],      0.0, 1.0),
        np.clip(raw_state["entry_age"],         0.0, 1.0),
        np.clip(raw_state["new_mac_rate"],      0.0, 1.0),
    ], dtype=np.float32)

def describe_state(state: np.ndarray) -> str:
    lines = []
    for name, val in zip(STATE_FEATURES, state):
        if val > 0.85:
            status = "CRITICAL"
        elif val > 0.6:
            status = "WARNING"
        else:
            status = "OK"
        lines.append(f"  {name:<22}: {val:.3f}  [{status}]")
    return "\n".join(lines)