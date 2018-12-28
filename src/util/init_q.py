import numpy as np


def initialize_q(numStates, numActions):
    data = np.zeros((numStates, numActions))
    data.fill(10)
    return data
