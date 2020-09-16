import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from scipy.stats import rankdata


class DDQN_PER:
    def __init__(self, env):
        self.epsilon = 1.0
        self.epsilon_discount_rate = 0.999
        self.epsilon_minimum = 0
        self.batch_size = 64
        self.minimum_data = 500
        self.discount_rate = 0.99
        self.alpha = 0.7
        self.beta = 0.5
        self.alpha_discount_rate = 0.99
        self.beta_discount_rate = 0.99
        self.target_fix = 0
        self.count = 0
        self.env = env
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = 0.005, momentum = 0.5)
        self.update()
        self.replay_buffer = deque(maxlen = 2000)
        self.rng = np.random.default_rng()
        self.step_count = 0

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape = self.env.observation_space.shape))
        model.add(tf.keras.layers.Dense(24, activation = "relu"))
        model.add(tf.keras.layers.Dense(24, activation = "relu"))
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation = "linear"))
        return model

    def grad(self, replays, weights, batch_size):
        states = tf.constant([replay[0] for replay in replays], dtype = tf.dtypes.float32)
        actions = tf.constant([replay[1] for replay in replays], dtype = tf.dtypes.int32)
        rewards = tf.constant([replay[2] for replay in replays], dtype = tf.dtypes.float32)
        next_states = tf.constant([replay[3] for replay in replays], dtype = tf.dtypes.float32)
        dones = tf.constant([replay[4] for replay in replays], dtype = tf.dtypes.float32)
        weights = tf.constant(weights, dtype = tf.dtypes.float32)
        with tf.GradientTape() as t:
            next_actions = tf.math.argmax(tf.stop_gradient(self.model(states)), axis = 1)
            mask1 = tf.one_hot(indices = next_actions, depth = self.env.action_space.n)
            action_values = tf.math.reduce_sum(tf.stop_gradient(self.target_model(next_states) * mask1), axis = 1)
            y_true = rewards + self.discount_rate * (action_values * dones)
            mask2 = tf.one_hot(indices = actions, depth = self.env.action_space.n)
            y_pred = tf.math.reduce_sum((self.model(states) * mask2), axis = 1)
            errors = y_true - y_pred
            abs_errors = tf.math.abs(errors)
            loss = tf.math.divide(tf.math.reduce_sum(tf.math.square(errors) * weights), batch_size)
        return t.gradient(loss, self.model.trainable_variables), abs_errors.numpy()

    def learn(self):
        self.step_count += 1
        size = len(self.replay_buffer)
        if (size < self.minimum_data):
            return
        rank = rankdata([replay[5] for replay in self.replay_buffer])
        rank[-1] = 0
        rank += 1
        p = np.power((1.0 / rank), self.alpha)
        P = p / np.sum(p)
        w = np.power((size * P), (1 - self.beta))
        w /= np.amax(w)
        indices = self.rng.choice(size, self.batch_size, replace = False, p = P)
        replays = []
        weights = []
        for index in indices:
            replays.append(self.replay_buffer[index])
            weights.append(w[index])
        grads, abs_errors = self.grad(replays, weights, self.batch_size)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        i = 0
        for index in indices:
            self.replay_buffer[index][5] = abs_errors[i]
            i += 1    
        if (self.epsilon >= self.epsilon_minimum):
            self.epsilon *= self.epsilon_discount_rate
        self.alpha *= self.alpha_discount_rate
        self.beta *= self.beta_discount_rate
        if (self.target_fix > 0):
            self.count += 1
            if (self.target_fix == self.count):
                self.update()
                self.count = 0

    def update(self):
        self.target_model.set_weights(self.model.get_weights())

    def take_action(self, state):
        if (random.random() > self.epsilon):
            return np.argmax(((self.model(tf.constant([state]))).numpy())[0, :])
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
            self.replay_buffer.append([state, action, reward, next_state, int(not done), 0])
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
    
    
env = gym.make("CartPole-v0")
agent = DDQN_PER(env)
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
