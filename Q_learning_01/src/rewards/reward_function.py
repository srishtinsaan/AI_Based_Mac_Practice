def get_reward(action, fill, pressure, age, rate):
    """
    Maps network states and actions to specific project outcomes.
    """
    rewards_map = {
        "correct_delivery": 1.0,         # Success: Learning when empty
        "flooding_avoided": 0.8,         # Success: Blocking during attack
        "preventive_removal": 0.3,       # Success: Maintenance
        "necessary_flooding": -0.1,      # Neutral: Default fallback
        "unnecessary_flooding": -1.0,    # Failure: Blocking good traffic
        "wrong_port_stale": -0.5,        # Failure: Bad maintenance
        "early_removal": -0.3,           # Failure: Evicting fresh entries
        "table_overflow": -0.7           # Failure: Learning when full
    }

    outcome = "necessary_flooding" # Default fallback

    # ---------------- LEARN (Action 0) ----------------
    if action == 0:
        if fill < 50: outcome = "correct_delivery"
        elif fill >= 95: outcome = "table_overflow"
        else: outcome = "necessary_flooding"

    # ---------------- EVICT (Action 1) ----------------
    elif action == 1:
        if 80 <= fill < 95 and (age.lower() in ["aging", "stale"]):
            outcome = "preventive_removal"
        elif fill < 50: outcome = "early_removal"
        else: outcome = "wrong_port_stale"

    # ---------------- BLOCK (Action 4) ----------------
    elif action == 4:
        if fill >= 95 and (pressure.lower() == "high" or rate.lower() == "high"):
            outcome = "flooding_avoided"
        else: outcome = "unnecessary_flooding"

    return rewards_map.get(outcome, 0.0)