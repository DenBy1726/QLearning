import numpy as np


class Agent:
    def __init__(self, point, w, h, numActions):
        self.point = point
        self.prev = point
        self.q = init_q(w, h, numActions)
        self.defaultPoint = point
        self.profit = 0

    def clear_point(self):
        self.point = self.defaultPoint
        self.prev = self.point

    def update_profit(self, value):
        self.profit += value

    def clear_profit(self):
        self.profit = 0

    def mean_profit(self, agent2):
        return (self.profit + agent2.profit) / 2

    def get_index(self, agent2):
        return self.point[0] * 16 * 9 * 16 + self.point[1] * 9 * 16 + agent2.point[0] * 16 + agent2.point[1]

    def get_prev_index(self, agent2):
        return self.prev[0] * 16 * 9 * 16 + self.prev[1] * 9 * 16 + agent2.prev[0] * 16 + agent2.prev[1]

    def step(self, action):
        self.prev = self.point
        if (action == 0):
            self.point = (self.point[0] - 1, self.point[1])
        if (action == 1):
            self.point = (self.point[0], self.point[1] + 1)
        if (action == 2):
            self.point = (self.point[0] + 1, self.point[1])
        if (action == 3):
            self.point = (self.point[0], self.point[1] - 1)
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
        # return np.append(allowedActions, 4)
        return allowedActions

    def resolve_actions2(self, matrix, action, agent2):
        distance = abs(self.point[0] - agent2.point[0]) + abs(self.point[1] - agent2.point[1])
        allowedActions = np.array([], dtype=int)
        if (distance != 1):
            allowedActions = self.resolve_actions(matrix)
        else:
            if (self.point[0] > 0 and action != 2):
                allowedActions = np.append(allowedActions, 0)
            if (self.point[1] < matrix.shape[1] - 1 and action != 3):
                allowedActions = np.append(allowedActions, 1)
            if (self.point[0] < matrix.shape[0] - 1 and action != 0):
                allowedActions = np.append(allowedActions, 2)
            if (self.point[1] > 0 and action != 1):
                allowedActions = np.append(allowedActions, 3)

        return allowedActions


def init_q(w, h, numActions):
    q_values = []
    for i in range(0, int(h)):
        for j in range(w):
            for u in range(0, h):
                for v in range(w):
                    q_values.append(
                        np.concatenate([[i, j, u, v], np.repeat(10, numActions).astype(np.float, copy=False)]))
    return q_values
