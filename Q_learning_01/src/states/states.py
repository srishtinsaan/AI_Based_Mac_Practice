def get_state(mac_fill, flood, traffic, vlan, age, mac_rate):
    """
    Summarizes all 6 network metrics into 3 discrete states .
    """
    # S3: Critical [cite: 145-158]
    # Triggered by high pressure, full table, OR suspicious new MAC rates.
    if mac_fill >= 100 or flood == "High" or age == "Stale" or mac_rate == "High":
        return 2 
        
    # S2: Moderate [cite: 132-144]
    # Triggered by nearing capacity or aging entries.
    elif mac_fill >= 80 or age == "Aging" or traffic == "High":
        return 1
        
    # S1: Normal [cite: 119-131]
    # Default stable state.
    else:
        return 0