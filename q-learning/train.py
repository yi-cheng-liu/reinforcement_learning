import numpy as np
from gridworld_maze import GridWorldMazeEnv
from q_learning_agent import QLearningAgent

if __name__ == "__main__":
    env = GridWorldMazeEnv(seed=0)
    agent = QLearningAgent(seed=0)
    
    print("Your actions during the last episode:")
    total_reward = 0.0
    num_ep = 2000
    rewards_q_learning = np.zeros((num_ep,))
    for ep in range(num_ep):
        state = env.reset()
        print(agent.q_table)
        for step in range(1000):
            action = int(agent.select_action(state))
            if ep == num_ep - 1:
                print(action, end=" ")
            next_state, reward, done = env.step(action)

            rewards_q_learning[ep] = rewards_q_learning[ep] + reward

            if done:
                agent.train(state, action, reward, next_state, 1)
            elif step == 999:
                agent.train(state, action, reward, next_state, -1)
            else:
                agent.train(state, action, reward, next_state, 0)

            if num_ep - ep <= 500:
                total_reward = total_reward + (0.9 ** step) * reward

            if done:
                break
            else:
                state = next_state

    total_reward = total_reward / 500
    print("")
    print("Your total reward averaged over the last %d episodes:\n%.3f" % (500, total_reward))
