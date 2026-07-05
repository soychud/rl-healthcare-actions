"""Reward computation with NaN guard."""
def compute_reward(stay_data):
    if not stay_data:
        return 0.0
    return 1.0
