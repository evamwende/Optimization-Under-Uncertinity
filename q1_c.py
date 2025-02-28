import gurobipy as gp
from gurobipy import GRB

# --- Data from Table 1 ---
scenarios = [
    {"probability": 0.4, "first_class": 25, "business_class": 60, "economy_class": 200},
    {"probability": 0.3, "first_class": 20, "business_class": 40, "economy_class": 180},
    {"probability": 0.2, "first_class": 10, "business_class": 25, "economy_class": 175},
    {"probability": 0.1, "first_class": 5, "business_class": 10, "economy_class": 150},
]

# --- Parameters ---
total_capacity = 200
economy_profit = 1
business_profit = 2
first_profit = 3

# --- Stochastic Program ---
def solve_stochastic_program():
    model = gp.Model("ScotRail_Stochastic")

    # --- First-Stage Decisions (Here-and-Now Variables) ---
    x_e = model.addVar(vtype=GRB.INTEGER, name="x_e")
    x_b = model.addVar(vtype=GRB.INTEGER, name="x_b")
    x_f = model.addVar(vtype=GRB.INTEGER, name="x_f")

    # --- Second-Stage Decisions (Recourse Variables for Each Scenario) ---
    y_e = {}
    y_b = {}
    y_f = {}

    for s_idx, scenario in enumerate(scenarios):
        y_e[s_idx] = model.addVar(vtype=GRB.INTEGER, name=f"y_e_{s_idx}")
        y_b[s_idx] = model.addVar(vtype=GRB.INTEGER, name=f"y_b_{s_idx}")
        y_f[s_idx] = model.addVar(vtype=GRB.INTEGER, name=f"y_f_{s_idx}")

    # --- Objective Function ---
    objective = gp.quicksum(
        scenarios[s_idx]["probability"] * (economy_profit * y_e[s_idx] + business_profit * y_b[s_idx] + first_profit * y_f[s_idx])
        for s_idx in range(len(scenarios))
    )
    model.setObjective(objective, GRB.MAXIMIZE)

    # --- Constraints ---
    # First-Stage Constraint: Total Train Capacity
    model.addConstr(x_e + 1.5 * x_b + 2 * x_f <= total_capacity, "capacity")

    # Second-Stage Constraints (for each scenario)
    for s_idx, scenario in enumerate(scenarios):
        # Sales Cannot Exceed Demand
        model.addConstr(y_e[s_idx] <= scenario["economy_class"], f"demand_e_{s_idx}")
        model.addConstr(y_b[s_idx] <= scenario["business_class"], f"demand_b_{s_idx}")
        model.addConstr(y_f[s_idx] <= scenario["first_class"], f"demand_f_{s_idx}")

        # Sales Cannot Exceed Allocated Seats
        model.addConstr(y_e[s_idx] <= x_e, f"allocation_e_{s_idx}")
        model.addConstr(y_b[s_idx] <= x_b, f"allocation_b_{s_idx}")
        model.addConstr(y_f[s_idx] <= x_f, f"allocation_f_{s_idx}")

    # --- Solve ---
    model.optimize()

    # --- Extract Results ---
    if model.status == GRB.OPTIMAL:
        solution = {
            "x_e": x_e.x,
            "x_b": x_b.x,
            "x_f": x_f.x,
            "objective_value": model.objVal,
        }
        return solution
    else:
        return None  # Return None if no solution found

# --- Expected Value Problem ---
def solve_expected_value_problem():
    model = gp.Model("ScotRail_Expected_Value")

    # --- Decision Variables ---
    x_e = model.addVar(vtype=GRB.INTEGER, name="x_e")
    x_b = model.addVar(vtype=GRB.INTEGER, name="x_b")
    x_f = model.addVar(vtype=GRB.INTEGER, name="x_f")

    # --- Calculate Expected Demand ---
    expected_demand_e = sum(scenario["probability"] * scenario["economy_class"] for scenario in scenarios)
    expected_demand_b = sum(scenario["probability"] * scenario["business_class"] for scenario in scenarios)
    expected_demand_f = sum(scenario["probability"] * scenario["first_class"] for scenario in scenarios)

    # --- Objective Function ---
    objective = (
        x_e * economy_profit
        + x_b * business_profit
        + x_f * first_profit
    )
    model.setObjective(objective, GRB.MAXIMIZE)

    # --- Constraints ---
    model.addConstr(x_e + 1.5 * x_b + 2 * x_f <= total_capacity, "capacity")
    model.addConstr(x_e <= expected_demand_e, "demand_e")
    model.addConstr(x_b <= expected_demand_b, "demand_b")
    model.addConstr(x_f <= expected_demand_f, "demand_f")

    # --- Solve ---
    model.optimize()

    # --- Extract Results ---
    if model.status == GRB.OPTIMAL:
        solution = {
            "x_e": x_e.x,
            "x_b": x_b.x,
            "x_f": x_f.x,
            "objective_value": model.objVal,
        }
        return solution
    else:
        return None  # Return None if no solution found

# --- Function to Evaluate EV Solution in the SP Model ---
def evaluate_ev_in_sp(ev_solution):
    model = gp.Model("Evaluate_EV")

    # --- Decision Variables ---
    y_e = {}
    y_b = {}
    y_f = {}

    for s_idx, scenario in enumerate(scenarios):
        y_e[s_idx] = model.addVar(vtype=GRB.INTEGER, name=f"y_e_{s_idx}")
        y_b[s_idx] = model.addVar(vtype=GRB.INTEGER, name=f"y_b_{s_idx}")
        y_f[s_idx] = model.addVar(vtype=GRB.INTEGER, name=f"y_f_{s_idx}")

    # --- Constraints ---
    for s_idx, scenario in enumerate(scenarios):
        # Sales Cannot Exceed Demand
        model.addConstr(y_e[s_idx] <= scenario["economy_class"], f"demand_e_{s_idx}")
        model.addConstr(y_b[s_idx] <= scenario["business_class"], f"demand_b_{s_idx}")
        model.addConstr(y_f[s_idx] <= scenario["first_class"], f"demand_f_{s_idx}")

        # Sales Cannot Exceed Allocated Seats
        model.addConstr(y_e[s_idx] <= ev_solution["x_e"], f"allocation_e_{s_idx}")
        model.addConstr(y_b[s_idx] <= ev_solution["x_b"], f"allocation_b_{s_idx}")
        model.addConstr(y_f[s_idx] <= ev_solution["x_f"], f"allocation_f_{s_idx}")

    # --- Objective Function ---
    objective = gp.quicksum(
        scenarios[s_idx]["probability"] * (economy_profit * y_e[s_idx] + business_profit * y_b[s_idx] + first_profit * y_f[s_idx])
        for s_idx in range(len(scenarios))
    )
    model.setObjective(objective, GRB.MAXIMIZE)

    # --- Solve ---
    model.optimize()

    # --- Extract Results ---
    if model.status == GRB.OPTIMAL:
        solution = {
            "objective_value": model.objVal,
        }
        return solution
    else:
        return None  # Return None if no solution found

# --- Wait-and-See Problem ---
def solve_wait_and_see_problem():
    expected_profit = 0

    for s_idx, scenario in enumerate(scenarios):
        model = gp.Model(f"Wait_and_See_{s_idx}")

        # --- Decision Variables ---
        x_e = model.addVar(vtype=GRB.INTEGER, name="x_e")
        x_b = model.addVar(vtype=GRB.INTEGER, name="x_b")
        x_f = model.addVar(vtype=GRB.INTEGER, name="x_f")

        # --- Constraints ---
        model.addConstr(x_e + 1.5 * x_b + 2 * x_f <= total_capacity, "capacity")
        model.addConstr(x_e <= scenario["economy_class"], "demand_e")
        model.addConstr(x_b <= scenario["business_class"], "demand_b")
        model.addConstr(x_f <= scenario["first_class"], "demand_f")

        # --- Objective Function ---
        objective = (
            x_e * economy_profit
            + x_b * business_profit
            + x_f * first_profit
        )
        model.setObjective(objective, GRB.MAXIMIZE)

        # --- Solve ---
        model.optimize()

        if model.status == GRB.OPTIMAL:
            expected_profit += scenario["probability"] * model.objVal
        else:
            return None  # Return None if no solution found

    return expected_profit

# --- Main ---
if __name__ == "__main__":
    # Solve Stochastic Program
    sp_solution = solve_stochastic_program()

    if sp_solution:
        # Solve Expected Value Problem
        ev_solution = solve_expected_value_problem()

        if ev_solution:
            # Evaluate EV in SP
            eev_solution = evaluate_ev_in_sp(ev_solution)

            if eev_solution:
                # Calculate VSS
                vss = sp_solution['objective_value'] - eev_solution['objective_value']

                # Solve Wait-and-See Problem
                ws_solution = solve_wait_and_see_problem()

                if ws_solution:
                    # Calculate EVPI
                    evpi = ws_solution - sp_solution['objective_value']

                    # Print only the required outputs
                    print("Optimal Solution:")
                    print(f"Economy seats: {sp_solution['x_e']}")
                    print(f"Business seats: {sp_solution['x_b']}")
                    print(f"First-class seats: {sp_solution['x_f']}")
                    print(f"Expected Profit: {sp_solution['objective_value']:.2f}")
                    print(f"VSS: {vss:.2f}")
                    print(f"EVPI: {evpi:.2f}")
                else:
                    print("Could not solve Wait-and-See Problem")
            else:
                print("Could not evaluate EV in SP")
        else:
            print("Could not solve Expected Value Problem")
    else:
        print("Could not solve Stochastic Program")