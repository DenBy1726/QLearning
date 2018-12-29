import numpy as np
import itertools


class Agent:
    def __init__(self, point, numStates, numActions):
        self.point = point
        self.q = init_q(numStates, numActions)
        self.defaultPoint = point

    def clear_point(self):
        self.point = self.defaultPoint

    def get_index(self):
        x = max(self.q[:, 0]) + 1
        y = max(self.q[:, 1]) + 1
        return x * self.point[1] + self.point[0]

    def step(self, action):
        if (action == 0):
            self.point = (self.point[0], self.point[1] - 1)
        if (action == 1):
            self.point = (self.point[0] + 1, self.point[1])
        if (action == 2):
            self.point = (self.point[0], self.point[1] + 1)
        if (action == 3):
            self.point = (self.point[0] - 1, self.point[1])
        return self.point

    def do_action(self, action):
        self.step(action)
        return self.point

    def resolve_actions(self, matrix):
        allowedActions = np.array([], dtype=int)
        if (self.point[0] > 0):
            allowedActions = np.append(allowedActions, 0)
        if (self.point[1] < matrix.shape[1] - 1):
            allowedActions = np.append(allowedActions, 1)
        if (self.point[0] < matrix.shape[0] - 1):
            allowedActions = np.append(allowedActions, 2)
        if (self.point[1] > 0):
            allowedActions = np.append(allowedActions, 3)
        # return allowedActions
        return np.append(allowedActions, 4)


def init_q(numStates, numActions):
    q_values = np.zeros((numStates, numActions))
    q_values.fill(10)
    q_values[:, 0] = list(range(0, matrix.shape[1])) * matrix.shape[0]
    q_values[:, 1] = list(
        itertools.chain.from_iterable(itertools.repeat(x, matrix.shape[0]) for x in range(0, matrix.shape[1])))
    q_values[:, 2] = list(range(0, matrix.shape[1])) * matrix.shape[0]
    q_values[:, 3] = list(
        itertools.chain.from_iterable(itertools.repeat(x, matrix.shape[0]) for x in range(0, matrix.shape[1])))
    return q_values


def egreedy_policy(agent1, agent2, actions, epsilon=0.1):
    # Get a random number from a uniform distribution between 0 and 1,
    # if the number is lower than epsilon choose a random action
    if np.random.random() < epsilon:
        return actions[np.random.choice(len(actions))]
    # Else choose the action with the highest value
    else:
        state = next(x for x in agent1.q if tuple(x[0:2]) == agent1.point and tuple(x[2:4] == agent2.point))[4:]
        return actions[np.argmax(state[actions])]


def get_profit(agent1, agent2, matrix):
    agent1_profit = matrix[agent1.point[1], agent1.point[0]]
    agent2_profit = matrix[agent2.point[1], agent2.point[0]]

    done = False
    if (min(agent1_profit, agent2_profit) == - 100 or max(agent1_profit, agent2_profit) == 100):
        done = True
    distance = abs(agent1.point[0] - agent2.point[0]) + abs(agent2.point[1] - agent2.point[1])
    if (distance > 2):
        done = True
        agent1_profit -= 100
        agent2_profit -= 100
    if (distance == 0):
        agent1_profit -= 1
        agent2_profit -= 1
    return (agent1_profit, agent2_profit, done)


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
agent1 = Agent((8, 0), numStates, numActions + 4)
agent2 = Agent((8, 1), numStates, numActions + 4)

# Iterate over 500 episodes
for _ in range(500):
    agent1.clear_point()
    agent2.clear_point()
    done = False

    agent1.get_index()

    # While episode is not over
    while not done:
        # Choose action
        actions1 = agent1.resolve_actions(matrix)
        actions2 = agent2.resolve_actions(matrix)

        action1 = egreedy_policy(agent1, agent2, actions1, epsilon=0.1)
        action2 = egreedy_policy(agent2, agent1, actions2, epsilon=0.1)

        agent1.do_action(action1)
        agent2.do_action(action2)

        (reward1,reward2,done) = get_profit(agent1,agent2,matrix)

        td_target1 = reward1 + 0.5 + np.max(agent1.q[agent1.get_index()])
        td_target2 = reward2 + 0.5 + np.max(agent2.q[agent2.get_index()])
        # Update q_values
        td_target = reward + gamma * np.max(q_values[next_state])
        td_error = td_target - q_values[state][action]
        q_values[state][action] += learning_rate * td_error
        # Update state
        state = next_state
