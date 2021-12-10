import numpy
import random
from collections import defaultdict

import numpy as np


def reshape_obs(observation):
    """
    Reshapes and 'discretizes' an observation for Q-table read/write
    Make sure the state space is not too large!

    :param observation: The to-be-reshaped/discretized observation. Contains the position of the
    'players', as well as the position and movement.
    direction of the ball.
    :return: The reshaped/discretized observation
    """

    # two decimals for position are enough
    # reshaped = [numpy.round(observation[0][i], 1) for i in range(4)]
    # horizontal position of agent is irrelevant (always the same)
    # del reshaped[1]

    pad_ball_rel = observation[0][0] - observation[0][2]  # negative if ball lower than middle of paddle

    reshaped = [
        0 if pad_ball_rel == 0 else int(pad_ball_rel / abs(pad_ball_rel)),
        numpy.round(observation[0][2], 1),
        numpy.round(observation[0][3], 1),
    ]

    # get index of ball direction (0 to 5)
    for i in range(6):
        if observation[0][4 + i] == 1:
            reshaped.append(i)

    # returned stringified
    return str(reshaped)


class Agent:
    """
    Skeleton q-learner agent that the students have to implement
    """

    def __init__(
            self, id, actions_n, obs_space_shape,
            gamma=0.6,  # pick reasonable values for all of these!
            epsilon=0.2,
            min_epsilon=0.01,
            epsilon_decay=0.97,
            alpha=0.1
    ):
        """
        Initiates the agent

        :param id: The agent's id in the game environment
        :param actions_n: The id of actions in the agent's action space
        :param obs_space_shape: The shape of the agents observation space
        :param gamma: Depreciation factor for expected future rewards
        :param epsilon: The initial/current exploration rate
        :param min_epsilon: The minimal/final exploration rate
        :param epsilon_decay: The rate of epsilon/exploration decay
        :param alpha: The learning rate
        """
        self.id = id
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.actions_n = actions_n
        self.obs_space_shape = obs_space_shape
        self.alpha = alpha
        self.q = defaultdict(lambda: numpy.zeros(self.actions_n))

    def determine_action_probabilities(self, observation):
        """
        A function that takes the state as an input and returns the probabilities for each
        action in the form of a numpy array of length of the action space.
        :param observation: The agent's current observation
        :return: The probabilities for each action in the form of a numpy
        array of length of the action space.
        """
        # TODO: implement this!
        obs_key = reshape_obs(observation)
        return self.q[obs_key]  # action_probabilities

    def act(self, observation):
        """
        Determines an action, given the current observation.
        :param observation: the agent's current observation of the state of
        the world
        :return: the agent's action
        """

        probs = self.determine_action_probabilities(observation)

        # explore if all null or with probability epsilon or return action with highest reward
        if not any(probs) or random.random() < self.epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(probs)  # index of action with best reward

        # decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        return action

    def update_history(
            self, observation, action, reward, new_observation
    ):
        """
        Updates the agent's Q-table

        :param observation: The observation *before* the action
        :param action: The action that has been executed
        :param reward: The reward the action has yielded
        :param new_observation: The observation *after* the action
        :return:
        """
        # counterfactual next action, to later backpropagate reward to current action
        new_obs_key = reshape_obs(new_observation)
        obs_key = reshape_obs(observation)
        max_action_key = numpy.argmax(self.q[new_obs_key])  # 0, 1 or 2

        td_delta = reward + self.gamma * self.q[new_obs_key][max_action_key] - self.q[obs_key][action]
        self.q[obs_key][action] += self.alpha * td_delta
