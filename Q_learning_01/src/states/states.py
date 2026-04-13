def get_state(mac_fill, pressure, age):
    if mac_fill >= 100 or pressure == "High":
        return 2  # S3: Critical [cite: 145-147]
    elif mac_fill >= 80 or age == "Aging":
        return 1  # S2: Moderate [cite: 132-133]
    return 0      # S1: Normal [cite: 119-120]