import gurobipy as gp
from gurobipy import GRB
from branch_and_bound import *
from cuts import *
from presolve import *
from primal_heuristics import *
from branching import *
from node_selection import *

def add_sos2_pwl_gurobi(model, name, x, y, x_breaks, y_breaks):
    if len(x_breaks) != len(y_breaks):
        raise ValueError("x_breaks e y_breaks devem ter o mesmo tamanho.")
    if len(x_breaks) < 2:
        raise ValueError("É preciso ter pelo menos 2 breakpoints.")
    for i in range(len(x_breaks) - 1):
        if x_breaks[i] >= x_breaks[i + 1]:
            raise ValueError("x_breaks devem ser estritamente crescentes.")

    n = len(x_breaks)

    lambdas = [
        model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"{name}_lambda_{i}")
        for i in range(n)
    ]
    segments = [
        model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f"{name}_seg_{j}")
        for j in range(n - 1)
    ]

    model.update()

    model.addConstr(gp.quicksum(lambdas) == 1.0, name=f"{name}_convexity")
    model.addConstr(x == gp.quicksum(x_breaks[i] * lambdas[i] for i in range(n)), name=f"{name}_x_link")
    model.addConstr(y == gp.quicksum(y_breaks[i] * lambdas[i] for i in range(n)), name=f"{name}_y_link")
    model.addConstr(gp.quicksum(segments) == 1.0, name=f"{name}_one_segment")

    model.addConstr(lambdas[0] <= segments[0], name=f"{name}_adj_0")
    for i in range(1, n - 1):
        model.addConstr(lambdas[i] <= segments[i - 1] + segments[i], name=f"{name}_adj_{i}")
    model.addConstr(lambdas[n - 1] <= segments[n - 2], name=f"{name}_adj_{n - 1}")

    return lambdas, segments

def main():
    model = gp.Model("sos2_quad")
    model.setParam("OutputFlag", 0)

    x = model.addVar(lb=0.0, ub=4.0, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVar(lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")

    add_sos2_pwl_gurobi(
        model,
        name="quad_pwl",
        x=x,
        y=y,
        x_breaks=[0.0, 1.0, 2.0, 3.0, 4.0],
        y_breaks=[0.0, 1.0, 4.0, 9.0, 16.0],
    )

    model.addConstr(x >= 2.3, name="x_min")
    model.setObjective(y, GRB.MINIMIZE)

    model.update()
    model.write("problems_nl/sos2_quad.mps")
    print("Arquivo gerado: problems_nl/sos2_quad.mps")

if __name__ == "__main__":
    main()

    model_path = "problems_nl/sos2_quad.mps"

    bb = BranchAndBound(
        model_path=model_path,
        strategy=BestFirst(),
        direction=BranchDirection.CEIL_FIRST,
        branch_rule=FirstFractionalRule(),
        presolver=None,
        heuristics=[],
        cutting_planes=[],
        verbose=True,
    )

    result = bb.solve()
    if result["status"] == "optimal":
        print(f"  Valor total: {result['obj_value']:.4f}")