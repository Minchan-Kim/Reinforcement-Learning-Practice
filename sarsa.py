from random import randrange, random, choice

class Environment:
    def __init__(self):
        self.states = [[(i, j) for j in range(5)] for i in range(5)]
        self.actions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def get_states(self):
        """Returns all states in the Gridworld."""
        return self.states

    def get_actions(self):
        """Returns all available actions."""
        return self.actions
        
    def get_reward(self, state, action):
        """Return a numerical reward according to current state and action."""
        next_state = self.get_next_state(state, action)
        if ((next_state[0] == 1) and (next_state[1] == 2)):
            return -1
        if ((next_state[0] == 2) and (next_state[1] == 1)):
            return -1
        if ((next_state[0] == 2) and (next_state[1] == 2)):
            return 1
        return 0

    def get_next_state(self, state, action):
        """Returns next state according to current state and action."""
        next_state = [(state[0] + action[0]), (state[1] + action[1])]
        if (next_state[0] < 0):
            next_state[0] = 0
        elif (next_state[0] > 4):
            next_state[0] = 4
        if (next_state[1] < 0):
            next_state[1] = 0
        elif (next_state[1] > 4):
            next_state[1] = 4
        return (next_state[0], next_state[1])


class Sarsa():
    def __init__(self, environment):
        self.environment = environment
        self.step_size = 0.01
        self.epsilon = 0.1
        self.q_table = [[[0.0] * 4 for _ in range(5)] for _ in range(5)]

    def update(self):
        """Generates an episode and updates action-value function table."""
        states = self.environment.get_states()
        actions = self.environment.get_actions()
        #state = (0, 0)
        # Initialize starting state to non-terminating one.
        state = (2, 2)
        while (state == (2, 2)):
            state = states[randrange(5)][randrange(5)]
        action = self.take_action(state)
        while (state != (2, 2)):
            reward = self.environment.get_reward(state, action)
            next_state = self.environment.get_next_state(state, action)
            next_action = self.take_action(next_state)
            temp = 0
            temp = self.step_size * (reward + 0.9 * self.q_table[next_state[0]][next_state[1]][actions.index(next_action)])
            temp += ((1 - self.step_size) * self.q_table[state[0]][state[1]][actions.index(action)])
            self.q_table[state[0]][state[1]][actions.index(action)] = temp
            state = next_state
            action = next_action

    def take_action(self, state):
        """Choose action given the state by using epsilon-greedy policy."""
        action_value = self.q_table[state[0]][state[1]]
        actions = self.environment.get_actions()
        # Derive an epsilon-greedy policy from the action-value function table.
        policy = [(self.epsilon / len(actions))] * 4
        max_action_value = max(action_value)
        #max_count = action_value.count(max_action_value)
        index_list = []
        for index, q in enumerate(action_value):
            if q == max_action_value:
                #policy[index] = ((1 - (self.epsilon / 4.0) * (4 - max_count)) / float(max_count))
                index_list.append(index)
        policy[choice(index_list)] = (1 - self.epsilon + self.epsilon / len(actions))
        # Choose action using the epsilon-greedy policy derived above.
        action_index = 3
        x = random()
        y = 0
        for index, p in enumerate(policy):
            y += p
            if x < y:
                action_index = index
                break
        return actions[action_index]

    def get_difference(self, table1, table2):
        """Returns 1-norm of a difference between the two action-value tables."""
        norm = 0
        for i in range(5):
            for j in range(5):
                for k in range(4):
                    norm += abs(table1[i][j][k] - table2[i][j][k])
        return norm

    def get_q_table(self):
        """Returns current action-value function table."""
        current_q_table = [[self.q_table[i][j][:] for j in range(5)] for i in range(5)]
        return current_q_table

    def print_q_table(self):
        """Prints action-value function value of all states.
        Order of actions : [up, down, left, right]
        """
        print("Action-Value Function Table")
        for row in self.q_table:
            lst = []
            for action_value in row:
                result = []
                for q in action_value:
                    result.append(round(q, 2))
                lst.append(result)
            print(lst)


def main():
    agent = Sarsa(Environment())
    old_q_table = agent.get_q_table()
    max_iteration = 100000
    min_iteration = 2000
    threshold = 10 ** (-6)
    for cnt in range(max_iteration):
        agent.update()
        new_q_table = agent.get_q_table()
        if (agent.get_difference(new_q_table, old_q_table) < threshold and cnt > min_iteration):
            print("Iteration {}".format(cnt + 1))
            break
        if (cnt == (max_iteration - 1)):
            print("Iteration Failed")
        old_q_table = new_q_table
    agent.print_q_table()


if __name__ == "__main__":
    main()
