import numpy as np
from gridworld_maze import GridWorldMazeEnv
from gridworld_agent import GridWorldAgent

if __name__ == "__main__":
    env = GridWorldMazeEnv(seed=0)
    agent = GridWorldAgent(env)
    
    V = agent.value_iteration()
    env.visualize(V)
    
    policy = agent.policy_iteration()
    print("Your actions during the last episode:")
    total_reward = 0.0
    num_ep = 500
    for ep in range(num_ep):
        state = env.reset()
        for step in range(1000):
            action = int(policy[state])
            if ep == num_ep - 1:
                print(action, end=" ")
            next_state, reward, done = env.step(action)
            total_reward = total_reward + (0.9 ** step) * reward
            if done:
                break
            else:
                state = next_state
    total_reward = total_reward / num_ep
    print("\nYour total reward averaged over %d episodes:\n%.3f" % (num_ep, total_reward))