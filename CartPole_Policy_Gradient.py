import gym
import tensorflow as tf
import numpy as np


class RTG_PG:
    def __init__(self, env):
        self.env = env
        self.state = tf.placeholder(tf.float32, shape = ([None] + list(self.env.observation_space.shape)))
        self.layer_1 = tf.layers.Dense(units = 32, activation = tf.math.tanh)
        self.layer_2 = tf.layers.Dense(units = self.env.action_space.n)
        self.logits = self.layer_2(self.layer_1(self.state))
        self.action = tf.squeeze(tf.random.categorical(self.logits, num_samples = 1), axis = [1])
        self.rtgs = tf.placeholder(tf.float32)
        self.actions = tf.placeholder(tf.int32)
        self.masks = tf.one_hot(indices = self.actions, depth = self.env.action_space.n)
        self.log_policy = tf.math.reduce_sum(tf.math.multiply(tf.nn.log_softmax(self.logits), self.masks), axis = 1)
        self.episodes = tf.placeholder(tf.float32)
        self.loss = - tf.math.divide(tf.math.reduce_sum(tf.math.multiply(self.log_policy, self.rtgs)), self.episodes)
        self.update = tf.train.AdamOptimizer(learning_rate = 1e-2).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def take_action(self, state):
        return self.sess.run(self.action, {self.state: np.array([state])})[0]

    def learn(self, batch_size = 5000):
        states = []
        actions = []
        rtgs = []
        eps_rewards = []
        timesteps = 0
        episodes = 0
        while True:
            state = self.env.reset()
            rewards = []
            eps_reward = 0
            done = False
            while not done:
                action = self.take_action(state)
                states.append(state)
                actions.append(action)
                state, reward, done, info = self.env.step(action)
                rewards.append(reward)
                eps_reward += reward
                timesteps += 1
            eps_rtgs = np.zeros_like(rewards)
            eps_rtgs[-1] = reward
            for i in reversed(range(len(rewards) - 1)):
                eps_rtgs[i] = rewards[i] + eps_rtgs[i + 1]
            rtgs += list(eps_rtgs)
            eps_rewards.append(eps_reward)
            episodes += 1
            if (timesteps >= batch_size):
                break
        self.sess.run(self.update, 
                      feed_dict = {self.state: np.array(states), 
                                   self.actions: np.array(actions), 
                                   self.rtgs: np.array(rtgs), 
                                   self.episodes: float(episodes)})
        return episodes, np.mean(eps_rewards)

    def run_episode(self, render = False):
        done = False
        state = self.env.reset()
        rewards = []
        while not done:
            if render:
                self.env.render()
            action = self.take_action(state)
            state, reward, done, info = self.env.step(action)
            rewards.append(reward)
        return rewards


def main():
    env = gym.make("CartPole-v1")
    agent = RTG_PG(env)
    reward_mean = 0
    episodes = 0
    while True:
        episodes, reward_mean = agent.learn(batch_size = 10000)
        print("Number of episodes: {:3}, Mean reward: {:6.2f}".format(episodes, reward_mean))
        if (reward_mean >= env.spec.reward_threshold):
            rewards = []
            for i in range(100):
                reward = sum(agent.run_episode())
                rewards.append(reward)
                print("Reward of episode {:3}: {}".format((i + 1), reward))
            if (np.mean(rewards) >= env.spec.reward_threshold):
                print("{} solved!".format(env.spec.id))
                break
    env.close()

if __name__ == "__main__":
    main()
