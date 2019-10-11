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


class PolicyIteration:
    def __init__(self, environment):
        self.environment = environment
        self.value_table = [[0.0] * 5 for _ in range(5)]
        self.policy_table = [[[0.25] * 4 for _ in range(5)] for _ in range(5)]

    def policy_evaluation(self):
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


agent = PolicyIteration(Environment())
while True:
    agent.policy_evaluation()
    agent.print_value()
    agent.policy_improvement()
    command = input("continue(Y/N)?: ")
    if (command == "N" or command == "n"):
        break
agent.print_policy()