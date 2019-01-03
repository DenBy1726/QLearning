import math

import matplotlib.pyplot as plt
import numpy as np

from src.manyagents.agent import Agent


def draw_matrix(matrix, agent1, agent2):
    fig = plt.figure(1)
    m_copy = np.copy(matrix)
    m_copy[agent1.point] = -45
    m_copy[agent2.point] = -65
    plt.imshow(m_copy, interpolation='nearest')
    plt.ion()
    plt.show()
    fig.canvas.flush_events()


def egreedy_policy(agent1, agent2, actions, epsilon=0.1):
    # Get a random number from a uniform distribution between 0 and 1,
    # if the number is lower than epsilon choose a random action
    if np.random.random() < epsilon:
        return actions[np.random.choice(len(actions))]
    # Else choose the action with the highest value
    else:
        state = agent1.q[agent1.get_index(agent2)][4:]
        return actions[np.argmax(state[actions])]


def get_profit(agent1, agent2, matrix):
    agent1_profit = matrix[agent1.point]
    agent2_profit = matrix[agent2.point]

    done = False
    if (min(agent1_profit, agent2_profit) == - 100 or max(agent1_profit, agent2_profit) == 100):
        done = True
    distance = abs(agent1.point[0] - agent2.point[0]) + abs(agent1.point[1] - agent2.point[1])
    if (distance > 2):
        done = True
        agent1_profit -= 100
        agent2_profit -= 100
    if (distance == 0):
        agent1_profit -= 1
        agent2_profit -= 1
    return (agent1_profit, agent2_profit, done)


def train(agent1, agent2, matrix):
    # Iterate over 500 episodes
    for _ in range(1000):
        agent1.clear_point()
        agent2.clear_point()
        done = False

        agent1.clear_profit()
        agent2.clear_profit()
        # agent1.q = agent2.q
        # While episode is not over
        while not done:
            eps = math.exp(-float(_) / 100)
            # Choose action
            actions1 = agent1.resolve_actions(matrix)
            action1 = egreedy_policy(agent1, agent2, actions1, epsilon=eps)
            actions2 = agent2.resolve_actions2(matrix, action1, agent1)
            action2 = egreedy_policy(agent2, agent1, actions2, epsilon=eps)

            agent1.do_action(action1)
            agent2.do_action(action2)

            (reward1, reward2, done) = get_profit(agent1, agent2, matrix)

            agent1.update_profit(reward1)
            agent2.update_profit(reward2)
            if (done == True):
                print(str(_) + ":" + str(agent1.mean_profit(agent2)))

            (index1, index2) = (agent1.get_index(agent2), agent2.get_index(agent1))
            (prev_index1, prev_index2) = (agent1.get_prev_index(agent2), agent2.get_prev_index(agent1))

            td_target1 = reward1 + 0.33 * np.max(agent1.q[index1])
            td_target2 = reward2 + 0.33 * np.max(agent2.q[index2])

            td_error1 = td_target1 - agent1.q[prev_index1][action1 + 4]
            td_error2 = td_target2 - agent2.q[prev_index2][action2 + 4]

            agent1.q[prev_index1][action1 + 4] += 0.65 * td_error1
            agent2.q[prev_index2][action2 + 4] += 0.65 * td_error2

            # agent2.q[prev_index1][action1 + 4] += 0.5 * td_error1
            # agent1.q[prev_index2][action2 + 4] += 0.5 * td_error2

            if (_ % 500 == 0):
                draw_matrix(matrix, agent1, agent2)


def run(matrix, agent1, agent2, iteration=5000):
    for t in range(1, iteration):
        actions1 = agent1.resolve_actions(matrix)
        action1 = egreedy_policy(agent1, agent2, actions1, epsilon=0)
        actions2 = agent2.resolve_actions2(matrix, action1, agent1)
        action2 = egreedy_policy(agent2, agent1, actions2, epsilon=0)

        agent1.do_action(action1)
        agent2.do_action(action2)

        (reward1, reward2, done) = get_profit(agent1, agent2, matrix)
        if (done):
            return

        draw_matrix(matrix, agent1, agent2)
