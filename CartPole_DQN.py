import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from google.colab import drive


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
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = self.step_size, momentum = 0.5)
        self.update()
        self.replay_buffer = deque(maxlen = 2000)
        self.step_count = 0
 
    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape = self.env.observation_space.shape))
        model.add(tf.keras.layers.Dense(24, activation = "relu"))
        model.add(tf.keras.layers.Dense(24, activation = "relu"))
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation = "linear"))
        return model

    def grad(self, replays):
        states = tf.constant([replay[0] for replay in replays], dtype = tf.dtypes.float32)
        actions = tf.constant([replay[1] for replay in replays], dtype = tf.dtypes.int32)
        rewards = tf.constant([replay[2] for replay in replays], dtype = tf.dtypes.float32)
        next_states = tf.constant([replay[3] for replay in replays], dtype = tf.dtypes.float32)
        dones = tf.constant([replay[4] for replay in replays], dtype = tf.dtypes.float32)
        with tf.GradientTape() as t:
            action_values = tf.math.reduce_max(tf.stop_gradient(self.target_model(next_states)), axis = 1)
            y_true = rewards + self.discount_rate * (action_values * dones)
            mask = tf.one_hot(indices = actions, depth = self.env.action_space.n)
            y_pred = tf.math.reduce_sum((self.model(states) * mask), axis = 1)
            loss = tf.keras.losses.MSE(y_true, y_pred)
        return t.gradient(loss, self.model.trainable_variables)
 
    def learn(self):
        self.step_count += 1
        if (len(self.replay_buffer) < self.minimum_data):
            return
        replays = random.sample(self.replay_buffer, self.batch_size)
        grads = self.grad(replays)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
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
            return np.argmax(((self.model(tf.constant([state]))).numpy())[0, :])
        else:
            return random.randrange(self.env.action_space.n)
 
    def run_episode(self, learn = True, render = False):
        state = self.env.reset()
        reward_total = 0
        temp = 0
        if not learn:
            temp = self.epsilon
            self.epsilon = 0
        while True:
            if render:
                self.env.render()
            action = self.take_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.append((state, action, reward, next_state, int(not done)))
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
 
    def save(self):
        drive.mount('/content/drive')
        self.model.save_weights('/content/drive/My Drive/CartPole-v0_DQN_weights.h5')
        drive.flush_and_unmount()
 
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
    #agent.save()
    env.close()
    
    
if __name__ == "__main__":
    main()
