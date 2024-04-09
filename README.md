# Reinforcement Learning Code Implementation

## Policy and Value Iteration

### Markov Decision Processes (MDPs)

In MDPs, **Controlled Markov Chains** allow for the manipulation of state-transition probabilities via actions. For a given action `u`, the transition probability from state `i` to `j` is denoted as `Pij(u)`. Introducing a **policy**, a function mapping states to actions, turns a controlled MDP into an uncontrolled Markov chain when the policy is applied, fixing the transition probabilities.

In the context of **infinite-horizon discounted dynamic programming (DP)**, the objective is to maximize the expected total discounted reward over an infinite timeline, expressed as:

$$
V(x) = max_μ E[r(x, μ(x)) + αV(x')]
$$

This equation relates the value function $V(x)$ for a state $x$ to the expected reward $r(x, μ(x))$ for taking action $μ(x)$ in state $x$, plus the discounted value of the next state $x'$, with $α$ as the discount factor.

The **value iteration algorithm** iteratively updates the value function $V(x)$ for each state $x$ in the state space $X$, aiming to find the optimal policy $μ$ that maximizes the expected total discounted reward. The update rule for the value function $Vk+1(x)$ in iteration $k+1$ is:

$$
V_{k+1}(x) ← max_u∈U [r̄(x, u) + α ∑_{x'} Px,x′(u)Vk(x')]
$$

Here, `r̄(x, u)` is the mean reward for taking action `u` in state `x`, and `Px,x′(u)` is the transition probability from `x` to `x'` under action `u`. The process iterates until the change in value function between two successive iterations is less than a specified threshold `ε`, indicating convergence to the optimal value function `V(x)`.

Once `V(x)` is determined, the optimal policy can be identified by choosing the action `u` that maximizes the sum of the immediate reward and the discounted value of the next state, formalized as:

$$
arg max_u [r̄(x, u) + α ∑_{x'} Px,x′(a)V(x')]
$$


This approach simplifies the policy space to focus on **Markov policies** and further to **stationary policies** for time-homogeneous Markov chains, where the policy does not change over time.

## Q-Learning

Q-learning is a model-free reinforcement learning method that allows an agent to learn the best actions to take in different states through trial and error, without needing prior knowledge. For example, teaching a robot how to navigate through a map, and the robot only have 4 steps to take, up ⬆️, right ➡️, down ⬇️, and left ⬅️.

By iteratively updating Q-values using the Bellman equation $
V_k(j) = \max_{i \in N_k(j)}(r_k(j, i) + V_{k+1}(i))$, Q-learning gradually converges to the optimal Q-function. This makes the agent to choose actions that maximize expected returns from any state.

**Update Formula**
Adjusting the Q-value towards the new estimate promises a higher return in the given state. Overtime, the action is the state will have a more rewarding action.
$$
Q(s, a) \leftarrow Q(s, a) + \beta_k [r + \alpha \max_{a'}Q(s', a') - Q(s, a)]
$$
**Temporal Difference(TD)**: The difference between the new estimate and old estimate
$$r + \alpha \max_{a'}Q(s', a') - Q(s, a)$$

- **Q-value $Q(s, a)$**: Estimated Q-value for action $u$ in state $x$

- **New Estimate $\alpha \max_{a'}Q(s', a')$**: A new estimate based on the received reward $r$ and the maximum estimated value of the next state $s'$

- **Learning Rate $\beta$**: Learning rate, determines how fast new information in Q-value updates

- **Discount Factor $\alpha$**: Weights future rewards against immediate rewards, usually sets at $0.9$

## Policy Gradient

## Deep Q-Networks (DQN)

## Actor-Critic Methods

## Monte Carlo Tree Search (MCTS)
