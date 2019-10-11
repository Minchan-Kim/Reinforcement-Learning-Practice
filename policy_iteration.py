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
        return next_state


class PolicyIteration:
    def __init__(self, environment):
        self.environment = environment
        self.value_table = [[0.0] * 5 for _ in range(5)]
        self.policy_table = [[[0.25] * 4 for _ in range(5)] for _ in range(5)]

    def policy_evaluation(self):
        """Do a policy evaluation once."""
        states = self.environment.get_states()
        actions = self.environment.get_actions()
        for row in states:
            for state in row:
                value = 0
                if (state == (2, 2)):
                    continue
                for action_index, action in enumerate(actions):
                    temp = 0
                    next_state = self.environment.get_next_state(state, action)
                    temp += self.policy_table[state[0]][state[1]][action_index]
                    temp *= (self.environment.get_reward(state, action) + 0.9 * self.value_table[next_state[0]][next_state[1]])
                    value += temp
                self.value_table[state[0]][state[1]] = value

    def policy_improvement(self):
        """Do a policy improvement once."""
        states = self.environment.get_states()
        actions = self.environment.get_actions()
        for row in states:
            for state in row:
                q = []
                for action in actions:
                    next_state = self.environment.get_next_state(state, action)
                    q.append(self.environment.get_reward(state, action) + 0.9 * self.value_table[next_state[0]][next_state[1]])
                max_q = max(q)
                cnt = q.count(max_q)
                for action_index in range(len(actions)):
                    if q[action_index] == max_q:
                        self.policy_table[state[0]][state[1]][action_index] = (1.0 / cnt)
                    else:
                        self.policy_table[state[0]][state[1]][action_index] = 0.0

    def get_difference(self, table1, table2):
        """Returns 1-norm of a difference between the two value tables."""
        norm = 0
        for i in range(5):
            for j in range(5):
                norm += abs(table1[i][j] - table2[i][j])
        return norm

    def get_value(self):
        """Returns current state-value function for all states."""
        current_value_table = [[self.value_table[i][j] for j in range(5)] for i in range(5)]
        return current_value_table

    def print_value(self):
        """Prints values of all states."""
        print("Value Table")
        for row in self.value_table:
            lst = []
            for value in row:
                lst.append(round(value, 2))
            print(lst)

    def print_policy(self):
        """Prints policy of all states.
        Prints 1 if an agent takes the action.
        Otherwise prints 0.
        Order of actions : [up, down, left, right]
        """
        print("Policy Table")
        for row in self.policy_table:
            lst = []
            for policy in row:
                max_p = max(policy)
                result = []
                for p in policy:
                    if (p == max_p):
                        result.append(1)
                    else:
                        result.append(0)
                lst.append(result)
            print(lst)


def main():
    print("Policy Iteration")
    agent = PolicyIteration(Environment())
    old_value = agent.get_value()
    threshold = 0.01
    max_iteration = 50
    for cnt in range(max_iteration):
        print("Iteration {}".format(cnt + 1))
        agent.policy_evaluation()
        agent.print_value()
        agent.policy_improvement()
        new_value = agent.get_value()
        if (agent.get_difference(new_value, old_value) < threshold):
            break
        old_value = new_value
        if (cnt == 49):
            print("Iteration Failed")
    agent.print_policy()


if __name__ == "__main__":
    main()
