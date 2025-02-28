

import data as Data
from gurobipy import Model, GRB, quicksum


# Load data from ReadData.py
cities = Data.cities  # set of cities
scenarios = Data.scenarios  # set of scenarios
theta = Data.theta  # unit cost of delivery to city n in the first stage
theta_s = Data.theta_prime  # unit cost of transportation between city n and center in the second stage
h = Data.h  # unit cost of unused inventory
g = Data.g  # unit cost of shortage
I = Data.I  # inventory of the center at the beginning
Yn = Data.Yn  # inventory of city n at the beginning
demand = Data.demand  # demand of city n under scenario k
prob = 1.0 / len(scenarios)  # probability of scenario k

# Create the Gurobi model
model = Model("TwoStageStochasticProgram")

# First-stage decision variables: x_n (amount of inventory allocated to city n)
x = model.addVars(cities, name="x", lb=0)

# Second-stage decision variables:
# u_n^k: transportation from center to city n under scenario k
# v_n^k: transportation from city n to center under scenario k
# z_n^k: unused inventory at city n under scenario k
# s_n^k: shortage at city n under scenario k
u = model.addVars(cities, scenarios, name="u", lb=0)
v = model.addVars(cities, scenarios, name="v", lb=0)
z = model.addVars(cities, scenarios, name="z", lb=0)
s = model.addVars(cities, scenarios, name="s", lb=0)

# Objective function: Minimize total cost
# First-stage cost: sum of theta_n * x_n
first_stage_cost = quicksum(theta[n] * x[n] for n in cities)

# Second-stage cost: expected value of transportation, shortage, and unused inventory costs
second_stage_cost = quicksum(
    prob * quicksum(
        theta_s[n] * (u[n, k] + v[n, k]) + h * z[n, k] + g * s[n, k] 
        for n in cities
    )
    for k in scenarios
)

# Set the objective
model.setObjective(first_stage_cost + second_stage_cost, GRB.MINIMIZE)

# First-stage constraint: Total allocated inventory cannot exceed available inventory
model.addConstr(quicksum(x[n] for n in cities) <= I, name="inventory_constraint")

# Second-stage constraints: Inventory balance for each city and scenario
for n in cities:
    for k in scenarios:
        model.addConstr(
            Yn[n] + x[n] + v[n, k] + s[n, k] == demand[n, k] + z[n, k] + u[n, k],
            name=f"balance_{n}_{k}"
        )

# Optimize the model
model.optimize()

# Output the results
if model.status == GRB.OPTIMAL:
    print("Optimal First-Stage Solution:")
    for n in cities:
        print(f"x[{n}] = {x[n].x}")

    print(f"\nOptimal Objective Value: {model.objVal}")
    print(f"Solution Time: {model.Runtime} seconds")
else:
    print("No optimal solution found.")