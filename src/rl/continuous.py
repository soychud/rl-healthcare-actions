"""Continuous dosage action space extension."""
import numpy as np
class ContinuousAction:
    def __init__(self, low, high):
        self.space = (low, high)
