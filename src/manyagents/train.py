import math
import operator
import random
import numpy as np

from src.util.draw import draw_matrix, update_matrix


def train(matrix, state, point=(0, 0), point2=(0, 1), iteration=1000, explorationConst=100):
    # (fig, ax) = draw_matrix(matrix, (s2, s1), state=state)
    gamma = 0.4
    log = []
    for t in range(1, iteration):
        ((s11, s12), (s21, s22)) = init(point, point2)
        value = 0
        done = False
        while not done:
            eps = math.exp(-float(t) / explorationConst)
            action = get_action(s11, s12, s21, s22, eps, matrix, state)
            action = get_action(s21, s22, s11, s12, eps, matrix, state)
            previous_state_index = get_index(s11, s12, matrix)
            previous_state_index = get_index(s21, s22, matrix)
            (prev_s11, prev_s12, prev_s21, prev_s22) = (s11, s12, s21, s22)

            (s11, s12) = get_next_state(s11, s12, action)
            (s21, s22) = get_next_state(s21, s22, action)

            state_index = get_index(s11, s12, matrix)
            state_index2 = get_index(s21, s22, matrix)

            profit = get_profit(s11, s12, matrix)
            if profit == -100 or profit == 100:
                done = True
            value += profit
            (index, future_best) = get_best(s11, s12, matrix, state[state_index])
            sample = profit + gamma * future_best
            update_weight(state, action, sample, previous_state_index)
        log.append(value)
        # update_matrix(fig, ax, matrix, (s2, s1), state=state, prev_point=(prev_s2, prev_s1))
    return (state, log)


def get_action(s11, s12, s21, s22, epsilon, matrix, state):
    actions = resolve_actions(s11, s12, matrix, state)
    randVal = random.random()
    if (randVal < epsilon):
        return actions[np.random.randint(0, len(actions), 1)[0]]
    index = get_index(s11, s12, matrix)
    index2 = get_index(s21, s22, matrix)
    index, value = get_best_by_actions(state[index][index2], actions)
    return index


def update_weight(state, current_action, sample, index):
    alpha = 0.5
    state[index, current_action] = state[index, current_action] + alpha * (
            sample - state[index, current_action])


def get_best_by_actions(state, actions):
    return max([(actions[i], x) for i, x in enumerate(np.array(state)[actions])], key=operator.itemgetter(1))


def get_best(s1, s2, matrix, state):
    actions = resolve_actions(s1, s2, matrix, state)
    return get_best_by_actions(state, actions)


def resolve_actions(s1, s2, matrix, state):
    allowedActions = np.array([], dtype=int)
    if (s2 > 0):
        allowedActions = np.append(allowedActions, 0)
    if (s1 < matrix.shape[1] - 1):
        allowedActions = np.append(allowedActions, 1)
    if (s2 < matrix.shape[0] - 1):
        allowedActions = np.append(allowedActions, 2)
    if (s1 > 0):
        allowedActions = np.append(allowedActions, 3)
    # return allowedActions
    return np.append(allowedActions, 4)


def get_next_state(s1, s2, action):
    if (action == 0):
        return (s1, s2 - 1)
    if (action == 1):
        return (s1 + 1, s2)
    if (action == 2):
        return (s1, s2 + 1)
    if (action == 3):
        return (s1 - 1, s2)
    return (s1, s2)


def get_index(s1, s2, matrix):
    return s1 * matrix.shape[0] + s2


def get_profit(s1, s2, matrix):
    return matrix[s2, s1]


def init(point, point1):
    return ((point[1], point[0]), (point1[1], point1[0]))


def run(matrix, state, point=(0, 0), iteration=5000):
    (s1, s2) = init(point)
    (fig, ax) = draw_matrix(matrix, (s2, s1), state=state)
    for t in range(1, iteration):
        action = get_action(s1, s2, 0, matrix, state)
        (prev_s1, prev_s2) = (s1, s2)
        (s1, s2) = get_next_state(s1, s2, action)
        profit = get_profit(s1, s2, matrix)
        if profit == -100:
            (s1, s2) = init(point)
        update_matrix(fig, ax, matrix, (s2, s1), state=state, prev_point=(prev_s2, prev_s1))
