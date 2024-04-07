import numpy as np

class QLearningAgent:

    def __init__(self, seed=None):
        self.q_table = np.zeros((64, 4))      # The Q table.
        self.learning_rate = 0.1              # Learning rate.
        self.learning_rate_decay = 0.998      # learning rate decay
        self.min_learning_rate = 0.001
        self.epsilon = 0.5                    # For the epsilon-greedy exploration
        self.epsilon_decay = 0.99             # epsilon decay
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def select_action(self, state):
        """
        This function returns an action for the agent to take.
        Args:
            state: the state in the current step
        Returns:
            action: the action that the agent plans to take in the current step
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice(4)
        else:
            action = np.argmax(self.q_table[state])

        return action

    def train(self, cur_state, cur_action, reward, next_state, done):
        """
        This function is used for the update of the Q table
        Args:
            - cur_state: the current state
            - cur_action: the current action
            - reward: the reward received
            - next_state: the next state observed
            - `done=1` means that the agent reaches the terminal state (`next_state=6`) and the episode terminates;
              `done=0` means that the current episode does not terminate;
              `done=-1` means that the current episode reaches the maximum length and terminates.
              We set the maximum length of each episode to be 1000.
        """
        if done == 1:
            temportal_difference = reward
        else:
            temportal_difference = reward + 0.9 * np.max(self.q_table[next_state]) - self.q_table[cur_state, cur_action]

        # Q-table update
        self.q_table[cur_state, cur_action] += self.learning_rate * temportal_difference

        # Update epsilon and learning rate
        if done != 0:
            self.learning_rate *= self.learning_rate_decay
            self.learning_rate = max(self.learning_rate, self.min_learning_rate)
            self.epsilon *= self.epsilon_decay
