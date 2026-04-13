class ActionSpace:
    """Action definitions for the switch controller [cite: 81-87]"""
    LEARN_MAC     = 0  # Add new MAC to table [cite: 88]
    EVICT_ENTRY   = 1  # Remove stale/old MAC [cite: 89]
    FLOOD         = 2  # Broadcast to all ports [cite: 90]
    ASSIGN_MAC    = 3  # Link MAC to specific VLAN/Port [cite: 91]
    BLOCK_PORT    = 4  # Security: Block suspicious port [cite: 92]
    UPDATE_CONFIG = 5  # Update TTL or Redis settings [cite: 93]

    @staticmethod
    def get_all_actions():
        return [0, 1, 2, 3, 4, 5]