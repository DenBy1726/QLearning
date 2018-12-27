from src.simple.train import train
from src.util.draw_matrix import draw_matrix
from src.util.init_q import initialize_q
import numpy as np

matrix = np.array([
    [-1, -1, -1, -1, -1, -1, -100, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -100, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -100, -1, -1, -1, -1, -1, -1, -100, -1, -1],
    [-1, -1, -1, -1, -1, -1, -100, -1, -1, -1, -1, -1, -1, -100, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -100, -1, -1],
    [-1, -1, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -1, 100]], np.int32)
initial = (8, 0)
numStates = matrix.shape[0] * matrix.shape[1]
numActions = 5

# draw_matrix(matrix, initial)
states = initialize_q(numStates, numActions)
train(matrix, states, initial)
print()
