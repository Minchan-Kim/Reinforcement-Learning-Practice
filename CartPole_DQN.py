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
        self.epsilon_discount_rate = 0.999
        self.epsilon_minimum = 0
        self.batch_size = 64
        self.minimum_data = 500
        self.discount_rate = 0.99
        self.step_size = 0.005
        self.target_fix = 0
        self.count = 0
        self.env = env
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update()
        self.replay_buffer = deque(maxlen = 2000)
        self.step_count = 0

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape = self.env.observation_space.shape))
        model.add(layers.Dense(24, activation = "relu"))
        model.add(layers.Dense(24, activation = "relu"))
        model.add(layers.Dense(self.env.action_space.n, activation = "linear"))
        model.compile(optimizer = keras.optimizers.SGD(learning_rate = self.step_size, momentum = 0.5), loss = "mse")
        return model

    def learn(self):
        self.step_count += 1
        if (len(self.replay_buffer) < self.minimum_data):
            return
        replays = random.sample(self.replay_buffer, self.batch_size)
        input_data = np.array([replay[0] for replay in replays])
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
        if (self.target_fix > 0):
            self.count += 1
            if (self.target_fix == self.count):
                self.update()
                self.count = 0

    def update(self):
        self.target_model.set_weights(self.model.get_weights())

    def take_action(self, state):
        if (random.random() > self.epsilon):
            return np.argmax(self.model.predict(np.array([state]), batch_size = 1)[0])
        else:
            return random.randrange(self.env.action_space.n)

    def run_episode(self, learn = True):
        state = self.env.reset()
        reward_total = 0
        temp = 0
        if not learn:
            temp = self.epsilon
            self.epsilon = 0
        while True:
            action = self.take_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.append((state, action, reward, next_state, done))
            reward_total += reward
            if learn:
                self.learn()
            if done:
                break
            state = next_state        
        if learn and (not (self.target_fix > 0)):
            self.update()
        if not learn:
            self.epsilon = temp
        return reward_total

    def get_step_count(self):
        return self.step_count


def main():
    env = gym.make("CartPole-v0")
    agent = DQN(env)
    max_iteration = 500
    max_episode = 100
    rewards = []
    reward = 0
    reward_threshold = ((env.spec.reward_threshold + env.spec.max_episode_steps) / 2.0)
    for i in range(max_iteration):
        reward = agent.run_episode()
        print("Reward of episode {:3}: {}".format((i + 1), reward))
        if (reward > reward_threshold):
            rewards.clear()
            for j in range(max_episode):
                reward = agent.run_episode(learn = False)
                rewards.append(reward)
                print("Reward of trial {:3}: {}".format((j + 1), reward))
                if (sum(rewards) < (max_episode * (env.spec.reward_threshold + 2 * j - 198))):
                    break
            if (sum(rewards) >= (env.spec.reward_threshold * max_episode)):
                print("CartPole-v0 solved in {} steps!".format(agent.get_step_count()))
                break
    env.close()

if __name__ == "__main__":
    main()
