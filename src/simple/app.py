from src.simple.train import train, run
from src.util.draw import draw_matrix, draw_plot
from src.util.init_q import initialize_q
import numpy as np
import json
import sys


def train_q():
    states = initialize_q((numStates, numActions))
    (states, log) = train(matrix, states, initial)
    draw_plot(log)
    json_states = json.dumps(states.tolist())
    with open('./result/q_states.json', 'w') as file:
        file.write(json_states)


def run_q():
    with open('./result/q_states.json') as f:
        data = json.load(f)
    states = np.array(data)

    run(matrix, states, initial)


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

if __name__ == "__main__":
    if (len(sys.argv) > 1 and sys.argv[1] == "train"):
        train_q()
    else:
        run_q()
