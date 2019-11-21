from __future__ import absolute_import, division, print_function, unicode_literals
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
from collections import deque


class DQN:
    def __init__(self, env):
        self.epsilon = 1.0
        self.epsilon_discount_rate = 0.99
        self.epsilon_minimum = 0.01
        self.batch_size = 64
        self.minimum_data = 100
        self.discount_rate = 0.99
        self.step_size = 0.001

        self.env = env
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.replay_buffer = deque(maxlen = 2000)

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape = self.env.observation_space.shape))
        model.add(layers.Dense(24, activation = "relu"))
        model.add(layers.Dense(24, activation = "relu"))
        model.add(layers.Dense(self.env.action_space.n, activation = "linear"))
        model.compile(optimizer = "sgd", loss = "mse")
        return model

    def learn(self):
        if (len(self.replay_buffer) < self.minimum_data):
            return
        replays = random.sample(self.replay_buffer, self.batch_size)
        input_data = []
        for replay in replays:
            input_data.append(replay[0])
        input_data = np.stack(input_data)
        target_data = self.model.predict(input_data)
        for i in range(self.batch_size):
            state, action, reward, next_state, done = replays[i]
            if not done:
                action_value = np.amax(self.target_model.predict(np.array([next_state]), batch_size = 1)[0])
                target_data[i, action] = reward + self.discount_rate * action_value
            else:
                target_data[i, action] = reward
        self.model.fit(input_data, target_data, batch_size = self.batch_size, epochs = 1, verbose = 0)
        if (self.epsilon >= self.epsilon_minimum):
            self.epsilon *= self.epsilon_discount_rate

    def update(self):
        self.target_model.set_weights(self.model.get_weights())

    def take_action(self, state):
        if (random.random() > self.epsilon):
            return np.argmax(self.model.predict(np.array([state]), batch_size = 1)[0])
        else:
            return random.randrange(self.env.action_space.n)

    def run_episode(self):
        state = self.env.reset()
        score = 0
        while True:
            action = self.take_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.append((state, action, reward, next_state, done))
            score += reward
            self.learn()
            if done:
                break
            state = next_state
        self.update()
        return score


def main():
    env = gym.make("CartPole-v0")
    agent = DQN(env)
    max_iteration = 100
    score = 0
    for i in range(max_iteration):
        score = agent.run_episode()
        print("Score of episode {}: {}".format((i + 1), score))
    env.close()

if __name__ == "__main__":
    main()