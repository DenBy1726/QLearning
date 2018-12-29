from src.manyagents.train import train, run
from src.util.draw import draw_matrix, draw_plot
from src.util.init_q import initialize_q
import numpy as np
import json

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
initial2 = (8, 1)
numStates = matrix.shape[0] * matrix.shape[1]
numActions = 5

states = initialize_q((numStates, numStates, numActions))
(states, log) = train(matrix, states, initial)

json_states = json.dumps(states.tolist())
with open('./result/q_states.json', 'w') as file:
    file.write(json_states)

with open('./result/q_states.json') as f:
    data = json.load(f)
states = np.array(data)

draw_plot(log)

run(matrix, states, initial)
print()
