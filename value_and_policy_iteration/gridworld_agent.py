import numpy as np
from gridworld_maze import GridWorldMazeEnv
env = GridWorldMazeEnv(seed=0)

class GridWorldAgent:
    def __init__(self, env, discount_factor=0.9):
        self.env = env
        self.alpha = discount_factor
        self.V = np.zeros((64,))
        self.policy = np.zeros((64,), dtype=int)

    def state_transition_func(self, state, action):
        # Wrapper for the environment's state transition function
        return self.env.state_transition_func(state, action)

    def value_iteration(self, threshold=0.01):
        """
        Please use value iteration to compute the optimal value function and store it into the vector V with size 64
        For example, V[0] means the optimal value function for state 0.
        """
        V = np.zeros((64,))
        theta = 0.01
        alpha = 0.9
        while True:
            delta = 0
            for cur_state in range(64):
                if cur_state == 6:  # Terminal state
                    continue
                v = V[cur_state]
                max_value = float('-inf') # find the max value in each action

                # Every State have 4 actions: up(0), right(1), down(2), left(3)
                for cur_action in range(4):
                    next_state = env.state_transition_func(cur_state, cur_action)
                    if (cur_state, cur_action) in [(5, 1), (7, 3), (14, 0)] and next_state == 6:
                        reward = 10
                    else:
                        reward = 0.5 * (-1) + 0.5 * (-2)
                    # Bellman Equation
                    value = reward + alpha * V[next_state]
                    max_value = max(max_value, value)
                    V[cur_state] = max_value
                    delta = max(delta, abs(v - V[cur_state]))
            if delta < theta:
                break
    
    def policy_v(self, state, V_star):
        """
        Implement the policy of the agent given the optimal value function
        Args:
            state: the current state of the agent, i.e, the grid where the agent is located.
            V_star: the optimal value function V*, a numpy array with size $64$
        Returns:
            action: the action that the agent plans to take.
                    The value of action should be 0, 1, 2, or 3, representing up, right, down, or left, respectively.
        """
        alpha = 0.9
        actions = [0, 1, 2, 3]  # Possible actions: up, right, down, left
        best_action = None
        best_value = float('-inf')

        for cur_action in actions:
            next_state = env.state_transition_func(state, cur_action)
            if (state, cur_action) in [(5, 1), (7, 3), (14, 0)] and next_state == 6:
                reward = 10
            else:
                reward = 0.5 * (-1) + 0.5 * (-2)

            # Bellman Equation
            value = reward + alpha * V_star[next_state]

            if value > best_value:
                best_value = value
                best_action = cur_action

        return best_action

    def policy_evaluation(self, mu):
        """
        Policy evaluation: calculate the value function of a given policy
        Args:
            mu: a given policy, which is a numpy array with size 64.
                Example: mu[s] = 0 or 1 or 2 or 3, which represents the action in state s
        Returns:
            V: the value function of the given policy mu, which is a numpy array with size $64$
        """
        V = np.zeros((64,))
        V_pre = np.zeros((64,))
        eps = 1e-3
        error = 100
        while error >= eps:
            for s in range(64):
                if s == 6:
                    V[s] = 0
                else:
                    a = mu[s]
                    if (s == 5 and a == 1) or (s == 7 and a == 3) or (s == 14 and a == 0):
                        V[s] = 10
                    else:
                        V[s] = -1.5 + 0.9 * V_pre[env.state_transition_func(s, a)]
            error = np.max(np.abs(V_pre - V))
            V_pre = V.copy()
        return V

    def policy_iteration(self):
        """
        Implement the policy iteration algorithm
        Returns:
            mu: the optimal policy obtained from the policy iteration algorithm.
                mu is a numpy array with size 64.
                mu[s] = 0 or 1 or 2 or 3, which represents the action in state s
        """
        mu = np.zeros((64,))
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        stable = False

        while not stable:
            V = self.policy_evaluation(mu)

            # Policy Improvement
            stable = True
            for cur_state in range(64):
                old_action = mu[cur_state]
                if cur_state == 6:
                    continue
                max_value = float('-inf')
                for cur_action in range(4):
                    next_state = env.state_transition_func(cur_state, cur_action)
                    reward = -1.5
                    if (cur_state, cur_action) in [(5, 1), (7, 3), (14, 0)] and next_state == 6:
                        reward = 10
                    value = reward + 0.9 * V[next_state]
                    if value > max_value:
                        max_value = value
                        mu[cur_state] = cur_action
                    if old_action != mu[cur_state]:
                        stable = False

        return mu