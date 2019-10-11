class Environment:
    def __init__(self):
        self.states = [[(i, j) for j in range(5)] for i in range(5)]
        self.actions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions
        
    def get_reward(self, state, action):
        next_state = self.get_next_state(state, action)
        if ((next_state[0] == 1) and (next_state[1] == 2)):
            return -1
        if ((next_state[0] == 2) and (next_state[1] == 1)):
            return -1
        if ((next_state[0] == 2) and (next_state[1] == 2)):
            return 1
        return 0

    def get_next_state(self, state, action):
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


class ValueIteration:
    def __init__(self, environment):
        self.environment = environment
        self.value_table = [[0.0] * 5 for _ in range(5)]
        self.policy_table = [[[0.25] * 4 for _ in range(5)] for _ in range(5)]

    def value_iteration(self):
        states = self.environment.get_states()
        actions = self.environment.get_actions()
        for row in states:
            for state in row:
                if (state == (2, 2)):
                    continue
                action_value = []
                for action in actions:
                    next_state = self.environment.get_next_state(state, action)
                    q = self.environment.get_reward(state, action) + 0.9 * self.value_table[next_state[0]][next_state[1]]
                    action_value.append(q)
                self.value_table[state[0]][state[1]] = max(action_value)

    def policy_determination(self):
        states = self.environment.get_states()
        actions = self.environment.get_actions()
        for row in states:
            for state in row:
                action_value = []
                for action in actions:
                    next_state = self.environment.get_next_state(state, action)
                    q = self.environment.get_reward(state, action) + 0.9 * self.value_table[next_state[0]][next_state[1]]
                    action_value.append(q)
                max_action_value = max(action_value)
                cnt = action_value.count(max_action_value)
                for index in range(len(actions)):
                    if (action_value[index] == max_action_value):
                        self.policy_table[state[0]][state[1]][index] = 1.0 / cnt
                    else:
                        self.policy_table[state[0]][state[1]][index] = 0

    def print_value(self):
        for row in self.value_table:
            lst = []
            for value in row:
                lst.append(round(value, 2))
            print(lst)

    def print_policy(self):
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


print("Value Iteration")
agent = ValueIteration(Environment())
while True:
    agent.value_iteration()
    agent.print_value()
    command = input("continue(Y/N)?: ")
    if (command == "N" or command == "n"):
        break
agent.policy_determination()
agent.print_policy()