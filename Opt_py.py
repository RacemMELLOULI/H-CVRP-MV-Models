import os
import sys
import subprocess

# Ensure the package 'cplex' is installed
try:
    import cplex
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cplex"])

# Ensure the package 'numpy' is installed
try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])

import cplex
import numpy as np

# Define the H-VRP problem data
n = 10  # Number of nodes (including depot as node 0)
q = [0, 3, 4, 2, 7, 5, 3, 2, 7, 2]  # Demand at each node (0 is the depot)
m = 4  # Number of available vehicles
Q = [10, 10, 15, 20]   # Vehicle capacity vector
c = np.array([
 [ 0.0,  8.9,  7.1, 10.4, 26.5, 15.3, 10.5, 39.4, 27.0, 13.8],
 [ 8.9,  0.0, 10.2, 19.2, 19.7,  9.4, 19.1, 35.0, 32.4, 18.4],
 [ 7.1, 10.2,  0.0, 12.3, 30.0, 19.3, 15.7, 33.0, 22.2, 20.9],
 [10.4, 19.2, 12.3,  0.0, 36.5, 25.3,  6.7, 44.2, 21.0, 17.0],
 [26.5, 19.7, 30.0, 36.5,  0.0, 11.2, 33.5, 47.0, 52.2, 26.0],
 [15.3,  9.4, 19.3, 25.3, 11.2,  0.0, 22.7, 42.2, 41.4, 17.0],
 [10.5, 19.1, 15.7,  6.7, 33.5, 22.7,  0.0, 48.7, 27.7, 10.8],
 [39.4, 35.0, 33.0, 44.2, 47.0, 42.2, 48.7,  0.0, 37.3, 52.7],
 [27.0, 32.4, 22.2, 21.0, 52.2, 41.4, 27.7, 37.3,  0.0, 37.9],
 [13.8, 18.4, 20.9, 17.0, 26.0, 17.0, 10.8, 52.7, 37.9,  0.0]], dtype=float)  # Ensure distances c are treated as floats

MAXV = 4  # Max Number of vehicles to use
MINV = 0  # Min Number of vehicles to use

# Display the problem data
print("Data for Capacitated Vehicle Routing Problem\n\twith Heterogenuous Fleet and Used Vehicle Number Restrictions : (H-CVRP-MV)\n")
print(f"(n) Number of nodes including depot: {n}")
print(f"(m) Number of available vehicles : {m}")
print("(q) Demand at each node:", q)
print(f"(Q) Vehicle capacity: {Q}")
print("(c) Distance or cost matrix:")
print("  ",np.array2string(c, separator='  ', prefix='  \t'))
print(f"(MAXV)\tNumber max of vehicles to use: {MAXV}")
print(f"(MINV)\tNumber min of vehicles to use: {MINV}")
print("\n--- Setting up the model ---\n")

# Define sets
N = range(n)       # Set of all nodes (including depot 0)
N1 = range(1, n)   # Set of customer nodes (excluding depot)
K = range(m)  # Set of vehicles

# Create the model
model = cplex.Cplex()
model.set_problem_type(cplex.Cplex.problem_type.MILP)
model.parameters.timelimit.set(3600)  # Set a time limit for solving (in seconds)

# Decision variables
x = {}  # x[(i, j, k)] binary variables for each pair (i, j) and each vehicle k
u = {}  # u[(i, k)] continuous variables for each customer i and each vehicle k, used for MTZ subtour elimination (delivered quantity after customer i) 

for k in K:
    for i in N:
        for j in N:
            if i != j:
                x[(i, j, k)] = f"x_{i}_{j}_{k}"
                model.variables.add(names=[x[(i, j, k)]], types=[model.variables.type.binary])
        if i > 0:
            u[(i, k)] = f"u_{i}_{k}"
            model.variables.add(names=[u[(i, k)]], lb=[q[i]], ub=[Q[k]], types=[model.variables.type.continuous])

# Objective function: min sum_{i,j in N, k in K} c[i][j] * x[(i,j,k)]
model.objective.set_linear([(x[(i, j, k)], c[i][j]) for i in N for j in N for k in K if i != j])
model.objective.set_sense(model.objective.sense.minimize)

# Constraints 1 for each used vehicle to start at the depot at most once: sum_{j in N1} x[0,j,k] = 1, for all k in K
for k in K:
    model.linear_constraints.add(
        lin_expr=[[ [x[(0, i, k)] for i in N1], [1] * (n - 1) ]],
        senses="L", rhs=[1]
    )

# Constraints 2 for Flow Balance for each customer and vehicle: sum_{i in N\{j}} x[(i,j,k)] = sum_{i in N\{j}} x[(j,i,k)], for all k in K, j in N
for k in K:
    for j in N:
        model.linear_constraints.add(
            lin_expr=[[ [x[(i, j, k)] for i in N if i != j] + [x[(j, i, k)] for i in N if i != j],
                       [1] * (n - 1) + [-1] * (n - 1) ]],
            senses="E", rhs=[0]
        )
# Constraints 3 for demand satisfaction (all customers are visited once): sum_{k in K, i in N\{j}} x[(i,j,k)] = 1, for all j in N1
for j in N1:
        model.linear_constraints.add(
            lin_expr=[[ [x[(i, j, k)] for k in K for i in N if i != j],
                       [1] * (n - 1)*m ]],
            senses="E", rhs=[1]
        )

# Constraints 4 for Vehcile Capacity restriction : sum_{for j in N1, i in N\{j}} x[(i,j,k)]*q[j] <= Q[k], for all k in K
for k in K:
        model.linear_constraints.add(
            lin_expr=[[ [x[(i, j, k)] for j in N1 for i in N if i != j],
                       [q[j] for j in N1 for i in N if i != j]]],
            senses="L", rhs=[Q[k]]
        )
# Constraints 5: MTZ constraints for subtour elimination: 
# u[(i,k)] - u[(j,k)] + Q[k] * x[(i, j, k)] <= Q[k] - q[j], for all i, j in N1, i != j, k in K
for k in K:
    for i in N1:
        for j in N1:
            if i != j:
                model.linear_constraints.add(
                    lin_expr=[[ [u[(i, k)], u[(j, k)], x[(i, j, k)]], [1, -1, Q[k]] ]],
                    senses="L", rhs=[Q[k] - q[j]]
                )
# u[(j,k)] + (Q[k] - q[j]) * x[(0, j, k)] <=  Q[k], for all j in N1, k in K
for k in K: 
    for j in N1:
                model.linear_constraints.add(
                    lin_expr=[[ [ u[(j, k)], x[(0, j, k)] ], [ 1, Q[k]-q[j] ] ]],
                    senses="L", rhs=[Q[k]]
                )

# Constraints 6 for Vehicle Use Restrictions (in Number) 
model.linear_constraints.add(
    lin_expr=[[ [ x[(0, i, k)] for k in K for i in N1 ], [1] * ((n - 1) * m) ]],
    senses="L", rhs=[MAXV]
)
model.linear_constraints.add(
    lin_expr=[[ [x[(0, i, k)] for k in K for i in N1], [1] * ((n - 1) * m) ]],
    senses="G", rhs=[MINV]
)

# Solve the model
model.solve()

# Retrieve and print the solution
solution = model.solution
print("\nObjective (Total distance):", solution.get_objective_value())
print("Solution status:", solution.get_status())

for k in K:
    print(f"\nRoute for vehicle {k}:\tQ[{k}]={Q[k]}")
    for i in N:
        for j in N:
            if i != j and solution.get_values(x[(i, j, k)]) > 0.5:
                if j>0: print(f"\tFrom {i} to {j}\tq[{j}]={q[j]}\tu[{j},{k}]={solution.get_values(u[(j, k)])}")
                else: print(f"\tFrom {i} to {j}")
