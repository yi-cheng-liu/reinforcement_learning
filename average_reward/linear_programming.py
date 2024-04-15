import numpy as np
from scipy.optimize import linprog

# Transition probabilities for each action
P1 = np.array([[0.5, 0.3, 0.2],
               [0.1, 0.4, 0.5],
               [0.3, 0.3, 0.4]])

P2 = np.array([[0.5, 0.3, 0.2],
               [0.5, 0.1, 0.4],
               [0.2, 0.5, 0.3]])

# Rewards for each state and action
r = np.array([[4, 2],
              [0, 1],
              [2, 3]])

# Utilities for each state and action
g = np.array([[1, 3],
              [2, 1],
              [3, 2]])


# Flatten the utilities matrix for the constraint
c = -r.flatten()

# Define the inequality constraints
A_ub = np.array([-g.flatten()])
b_ub = np.array([-1.5])

A_eq = np.array([
        [          1,           1,           1,           1,           1,           1],
        [1 - P1[0,0], 1 - P2[0,0],    -P1[1,0],    -P2[1,0],    -P1[2,0],    -P2[2,0]],
        [   -P1[0,1],    -P2[0,1], 1 - P1[1,1], 1 - P2[1,1],    -P1[2,1],    -P2[2,1]],
        [   -P1[0,2],    -P2[0,2],    -P1[1,2],    -P2[1,2], 1 - P1[2,2], 1 - P2[2,2]]])

b_eq = np.array([1, 0, 0, 0])

# Define the variable bounds (none of the probabilities can be negative)
bounds = [(0, 1)] * len(c)  # Six actions (states and actions combinations)

# Solve the linear programming problem
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

print("Status:", res.message)
if res.success:
    print("Optimal policy distribution:", res.x)
    print("Maximum average reward under constraints:", -res.fun)
else:
    print("Failed to find a feasible solution")