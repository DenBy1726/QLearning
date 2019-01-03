import json
import sys
import numpy as np

from src.manyagents.agent import Agent
from src.manyagents.train import train, run


def train_q():
    train(agent1, agent2, matrix)
    json_states = json.dumps((np.array(agent1.q).tolist(), np.array(agent2.q).tolist()))
    with open('./result/q_states.json', 'w') as file:
        file.write(json_states)


def run_q():
    with open('./result/q_states.json') as f:
        (q1, q2) = json.load(f)
        agent1.q = np.array(q1)
        agent2.q = np.array(q2)
    run(matrix, agent1, agent2)


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

numStates = matrix.shape[0] * matrix.shape[1]
numActions = 5

# Initialize Q arbitrarily, in this case a table full of zeros
agent1 = Agent((8, 0), 16, 9, numActions)
agent2 = Agent((8, 1), 16, 9, numActions)

if __name__ == "__main__":
    if (len(sys.argv) > 1 and sys.argv[1] == "train"):
        train_q()
    else:
        run_q()
