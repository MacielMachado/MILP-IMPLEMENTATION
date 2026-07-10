import math
import random
import time
import json
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

import gurobipy as gp
from gurobipy import GRB
import numpy as np

from node_selection import *
from branching import *

class BranchAndBound:
    def __init__(
        self,
        model_path: str,
        strategy: NodeSelection,
        direction: BranchDirection = BranchDirection.CEIL_FIRST,
        heuristics: list = None,
        cutting_planes: list = None,          
        presolver: "Presolver" = None,
        verbose: bool = True,
        branch_rule: "BranchVariableRule" = None
    ):
        self.strategy       = strategy
        self.direction      = direction
        self.heuristics     = heuristics or []
        self.cutting_planes = cutting_planes or []   
        self.presolver      = presolver
        self.verbose        = verbose
        self.branch_rule    = branch_rule or FirstFractionalRule()

        self.best_obj        = -float("inf")
        self.best_sol        = None
        self.nodes_explored  = 0
        self.nodes_pruned    = 0
        self.heuristic_calls = 0
        self.heuristic_hits  = 0
        self.tol             = 1e-6
        self.nodes_created   = 1
        self.pruned_by_infeasibility = 0
        self.pruned_by_bound = 0
        self.integer_nodes = 0
        self.max_depth = 0
        self.depth_histogram = defaultdict(int)
        self.max_open_nodes = 0
        self.branching_log = []
        self.branch_rule_calls = 0
        self.branch_rule_time = 0.0
        self.heuristic_stats = defaultdict(lambda: {
            "calls": 0,
            "solutions_found": 0,
            "improvements": 0,
            "time": 0.0,
            "depths": [],
            "best_improvement": 0.0,
        })

        self.cut_stats = defaultdict(lambda: {
            "calls": 0,
            "cuts_added": 0,
            "time": 0.0,
            "depths": [],
        })

        self.start_time = None
        self.lp_time = 0.0
        self.solve_start = None
        self.next_node_id = 0



        self.stats = {
            "instance": {},
            "presolve": {},
            "root_lp": {},
            "tree": {},
            "branching": {},
            "heuristics": {},
            "cuts": {},
            "timeline": [],
            "incumbents": [],
            "bounds_progress": [],
            "final": {}
        }

        base_model = gp.read(model_path)
        base_model.setParam("OutputFlag", 0)

        constrs = base_model.getConstrs()
        vars_list = base_model.getVars()
        A = base_model.getA()

        n_bin = sum(1 for v in vars_list if v.VType == GRB.BINARY)
        n_int = sum(1 for v in vars_list if v.VType == GRB.INTEGER)
        n_cont = sum(1 for v in vars_list if v.VType == GRB.CONTINUOUS)
        n_cons = len(constrs)
        n_vars = len(vars_list)
        nnz = A.nnz
        density = nnz / (n_vars * n_cons) if n_vars > 0 and n_cons > 0 else 0.0

        n_le = sum(1 for c in constrs if c.Sense == "<")
        n_ge = sum(1 for c in constrs if c.Sense == ">")
        n_eq = sum(1 for c in constrs if c.Sense == "=")

        self.stats["instance"] = {
            "model_path": model_path,
            "n_vars": n_vars,
            "n_constraints": n_cons,
            "n_binary": n_bin,
            "n_integer": n_int,
            "n_continuous": n_cont,
            "objective_sense": "MIN" if base_model.ModelSense == GRB.MINIMIZE else "MAX",
            "nnz": nnz,
            "density": density,
            "n_le_constraints": n_le,
            "n_ge_constraints": n_ge,
            "n_eq_constraints": n_eq,
        }


        if self.presolver is not None:
            feasible, presolve_history = self.presolver.run(base_model)
            self.stats["presolve"]["rules"] = presolve_history
            self.stats["presolve"]["summary"] = {
                "n_rules": len(presolve_history),
                "total_bounds_tightened": sum(r["bounds_tightened"] for r in presolve_history),
                "total_rows_removed": sum(r["rows_removed"] for r in presolve_history),
                "total_vars_fixed": sum(r["vars_fixed"] for r in presolve_history),
                "total_coefficients_changed": sum(r["coefficients_changed"] for r in presolve_history),
                "total_runtime": sum(r["runtime"] for r in presolve_history),
            }
            if not feasible:
                raise ValueError("Presolver detectou problema inviável antes do B&B.")

        self._sense = base_model.ModelSense
        if self._sense == GRB.MINIMIZE:
            self.best_obj = float("inf")

        vars_list        = base_model.getVars()
        self._n          = len(vars_list)
        self._names      = [v.VarName for v in vars_list]
        self._orig_vtype = [v.VType   for v in vars_list]
        self._orig_lb    = [v.LB      for v in vars_list]
        self._orig_ub    = [v.UB      for v in vars_list]

        self._integer_vars = {
            i for i, vt in enumerate(self._orig_vtype)
            if vt in (GRB.BINARY, GRB.INTEGER)
        }

        for i, v in enumerate(vars_list):
            if self._orig_vtype[i] in (GRB.BINARY, GRB.INTEGER):
                v.VType = GRB.CONTINUOUS
                if self._orig_vtype[i] == GRB.BINARY:
                    v.LB = max(v.LB, 0.0)
                    v.UB = min(v.UB, 1.0)
        base_model.update()
        self._base_model = base_model

    def _solve_lp(self, node: Node) -> str:
        model = self._base_model.copy()
        model.setParam("OutputFlag", 0)
        vars_list = model.getVars()

        for i, v in enumerate(vars_list):
            if i in node.lower_bounds:
                v.LB = max(v.LB, node.lower_bounds[i])
            if i in node.upper_bounds:
                v.UB = min(v.UB, node.upper_bounds[i])

        model.update()
        t0 = time.perf_counter()
        model.optimize()
        self.lp_time += time.perf_counter() - t0

        if model.Status == GRB.OPTIMAL:
            node.lp_bound    = model.ObjVal
            node.lp_solution = {i: vars_list[i].X for i in range(self._n)}
            return "optimal"
        elif model.Status == GRB.INFEASIBLE:
            return "infeasible"
        else:
            return "unbounded"

    def _branch(self, node: Node, var_idx: int) -> tuple:
        val       = node.lp_solution[var_idx]
        floor_val = math.floor(val)
        ceil_val  = math.ceil(val)

        child_floor = Node(
            node_id=self.next_node_id,
            parent_id=node.node_id,
            lower_bounds=dict(node.lower_bounds),
            upper_bounds=dict(node.upper_bounds),
            depth=node.depth + 1,
            parent_branch_var=var_idx,
            parent_branch_side="floor",
            parent_lp_bound=node.lp_bound,
            parent_lp_value=val,
        )
        self.next_node_id += 1
        child_floor.upper_bounds[var_idx] = floor_val

        child_ceil = Node(
            node_id=self.next_node_id,
            parent_id=node.node_id,
            lower_bounds=dict(node.lower_bounds),
            upper_bounds=dict(node.upper_bounds),
            depth=node.depth + 1,
            parent_branch_var=var_idx,
            parent_branch_side="ceil",
            parent_lp_bound=node.lp_bound,
            parent_lp_value=val,
        )
        self.next_node_id += 1
        child_ceil.lower_bounds[var_idx] = ceil_val
        self.nodes_created += 2
        return child_ceil, child_floor

    def _try_update_incumbent(self, solution: dict, source: str) -> bool:
        if solution is None:
            return False
        obj = sum(
            self._base_model.getVars()[i].Obj * solution[i]
            for i in range(self._n)
        )
        if self._is_better(obj, self.best_obj):
            self.best_obj = obj
            self.best_sol = {self._names[i]: solution[i] for i in range(self._n)}
            self.stats["incumbents"].append({
                "time": self._elapsed_time(),
                "node": self.nodes_explored,
                "objective": self.best_obj,
                "source": source,
            })
            self._log(f"    [{source}] novo incumbente heurístico: {self.best_obj:.4f}")
            return True
        return False

    def _run_heuristics(self, node: Node) -> None:
        for h in self.heuristics:
            hname = repr(h)
            self.heuristic_calls += 1
            self.heuristic_stats[hname]["calls"] += 1
            self.heuristic_stats[hname]["depths"].append(node.depth)

            old_best = self.best_obj
            t0 = time.perf_counter()
            solution = h.try_run(node, self)
            elapsed = time.perf_counter() - t0
            self.heuristic_stats[hname]["time"] += elapsed

            if solution is not None:
                self.heuristic_hits += 1
                self.heuristic_stats[hname]["solutions_found"] += 1
                improved = self._try_update_incumbent(solution, hname)
                if improved:
                    self.heuristic_stats[hname]["improvements"] += 1
                    if self._sense == GRB.MINIMIZE:
                        imp = old_best - self.best_obj
                    else:
                        imp = self.best_obj - old_best
                    self.heuristic_stats[hname]["best_improvement"] = max(
                        self.heuristic_stats[hname]["best_improvement"], imp
                    )

    def _run_cutting_planes(self, node: Node) -> int:
        total_added = 0
        for cp in self.cutting_planes:
            cpname = repr(cp)
            self.cut_stats[cpname]["calls"] += 1
            self.cut_stats[cpname]["depths"].append(node.depth)

            t0 = time.perf_counter()
            added = cp.try_run(node, self)
            elapsed = time.perf_counter() - t0

            self.cut_stats[cpname]["time"] += elapsed
            self.cut_stats[cpname]["cuts_added"] += added
            total_added += added
        return total_added

    def _is_better(self, new_val: float, current_best: float) -> bool:
        if self._sense == GRB.MINIMIZE:
            return new_val < current_best - 1e-6
        return new_val > current_best + 1e-6

    def _can_prune(self, lp_bound: float) -> bool:
        if self._sense == GRB.MINIMIZE:
            return lp_bound >= self.best_obj - 1e-6
        return lp_bound <= self.best_obj + 1e-6

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
            
    def _is_integral_value(self, x: float, tol: float = 1e-6) -> bool:
        return abs(x - round(x)) <= tol
    
    def _count_fractionals(self, node: Node) -> int:
        return sum(
            1 for j in self._integer_vars
            if j in node.lp_solution and not self._is_integral_value(node.lp_solution[j])
        )

    def _fractional_list(self, node: Node) -> list:
        return [
            {
                "var_idx": j,
                "var_name": self._names[j],
                "value": node.lp_solution[j],
                "fractionality": abs(node.lp_solution[j] - round(node.lp_solution[j]))
            }
            for j in self._integer_vars
            if j in node.lp_solution and not self._is_integral_value(node.lp_solution[j])
        ]

    def _elapsed_time(self) -> float:
        return time.perf_counter() - self.start_time if self.start_time is not None else 0.0

    def _evaluate_branch_side(self, node: "Node", var_idx: int, side: str) -> tuple[str, Optional[float]]:
        model = self._base_model.copy()
        model.setParam("OutputFlag", 0)
        vars_list = model.getVars()

        for i, v in enumerate(vars_list):
            if i in node.lower_bounds:
                v.LB = max(v.LB, node.lower_bounds[i])
            if i in node.upper_bounds:
                v.UB = min(v.UB, node.upper_bounds[i])

        x = node.lp_solution[var_idx]
        if side == "floor":
            vars_list[var_idx].UB = min(vars_list[var_idx].UB, math.floor(x))
        elif side == "ceil":
            vars_list[var_idx].LB = max(vars_list[var_idx].LB, math.ceil(x))
        else:
            raise ValueError(f"side inválido: {side}")

        model.update()
        t0 = time.perf_counter()
        model.optimize()
        self.lp_time += time.perf_counter() - t0

        if model.Status == GRB.OPTIMAL:
            return "optimal", model.ObjVal
        elif model.Status == GRB.INFEASIBLE:
            return "infeasible", None
        else:
            return "unbounded", None

    def _branch_gain(
        self,
        parent_bound: float,
        child_status: str,
        child_bound: Optional[float],
        infeasible_gain: float = 1e6,
    ) -> float:
        if child_status == "infeasible":
            return infeasible_gain

        if child_status != "optimal" or child_bound is None:
            return 0.0

        if self._sense == GRB.MINIMIZE:
            return child_bound - parent_bound
        else:
            return parent_bound - child_bound
        
    def _update_pseudocost_from_solved_node(self, node: "Node") -> None:
        if not isinstance(self.branch_rule, PseudoCostRule):
            return

        j = getattr(node, "parent_branch_var", None)
        side = getattr(node, "parent_branch_side", None)
        parent_bound = getattr(node, "parent_lp_bound", None)
        parent_x = getattr(node, "parent_lp_value", None)

        if j is None or side is None or parent_bound is None or parent_x is None:
            return

        gain = self._branch_gain(parent_bound, "optimal", node.lp_bound)
        if side == "floor":
            delta = parent_x - math.floor(parent_x)
            if delta > self.tol:
                pc = gain / delta
                self.branch_rule.down_sum[j] = self.branch_rule.down_sum.get(j, 0.0) + pc
                self.branch_rule.down_count[j] = self.branch_rule.down_count.get(j, 0) + 1
        elif side == "ceil":
            delta = math.ceil(parent_x) - parent_x
            if delta > self.tol:
                pc = gain / delta
                self.branch_rule.up_sum[j] = self.branch_rule.up_sum.get(j, 0.0) + pc
                self.branch_rule.up_count[j] = self.branch_rule.up_count.get(j, 0) + 1

    def solve(self) -> dict:
        sense_str = "MIN" if self._sense == GRB.MINIMIZE else "MAX"
        self.start_time = time.perf_counter()
        self.solve_start = self.start_time
        self._log(f"\n{'='*60}")
        self._log(f"  Branch & Bound — estratégia: {self.strategy}")
        self._log(f"  Direção: {self.direction.value} | Sentido: {sense_str}")
        self._log(f"  Branching: {self.branch_rule}")
        self._log(f"  Variáveis: {self._n} ({len(self._integer_vars)} inteiras)")
        self._log(f"  Heurísticas: {[repr(h) for h in self.heuristics]}")
        self._log(f"{'='*60}")

        open_nodes = self.strategy.initialize()
        root = Node(node_id=self.next_node_id)
        self.next_node_id += 1
        self.strategy.add_node(open_nodes, root)

        while open_nodes:
            node = self.strategy.select_node(open_nodes, self._sense)
            self.nodes_explored += 1

            self.max_depth = max(self.max_depth, node.depth)
            self.depth_histogram[node.depth] += 1
            self.max_open_nodes = max(self.max_open_nodes, len(open_nodes))

            node_record = {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "depth": node.depth,
                "open_nodes_before": len(open_nodes),
                "time": self._elapsed_time(),
            }

            status = self._solve_lp(node)

            if status == "optimal":
                self._update_pseudocost_from_solved_node(node)
        
            if status == "infeasible":
                self.nodes_pruned += 1
                self.pruned_by_infeasibility += 1
                node_record["status"] = "infeasible"
                node_record["action"] = "pruned"
                node_record["pruned_reason"] = "infeasible"
                self.stats["timeline"].append(node_record)
                self._log(
                    f"  Nó {self.nodes_explored:4d} (prof={node.depth:2d}) "
                    f"| INVIÁVEL — podado"
                )
                continue

            if status == "unbounded":
                node_record["status"] = "unbounded"
                node_record["action"] = "skipped"
                self.stats["timeline"].append(node_record)
                self._log(
                    f"  Nó {self.nodes_explored:4d} (prof={node.depth:2d}) "
                    f"| LP UNBOUNDED"
                )
                continue

            node_record["status"] = "optimal"
            node_record["lp_bound"] = node.lp_bound
            node_record["n_fractional"] = self._count_fractionals(node)

            self.stats["bounds_progress"].append({
                "time": self._elapsed_time(),
                "node": self.nodes_explored,
                "depth": node.depth,
                "lp_bound": node.lp_bound,
                "open_nodes": len(open_nodes),
                "best_obj": self.best_obj,
            })

            if node.depth == 0 and "lp_value" not in self.stats["root_lp"]:
                self.stats["root_lp"] = {
                    "lp_value": node.lp_bound,
                    "n_fractional": self._count_fractionals(node),
                    "fractional_vars": self._fractional_list(node),
                }

        
            if self.cutting_planes:
                n_cuts = self._run_cutting_planes(node)
                if n_cuts > 0:
                    self._log(f"    [cuts] {n_cuts} corte(s) adicionado(s) ao modelo base")
                    status = self._solve_lp(node)

                    if status == "optimal" and node.depth == 0:
                        self.stats["root_lp"]["lp_value_after_cuts"] = node.lp_bound
                        self.stats["root_lp"]["n_fractional_after_cuts"] = self._count_fractionals(node)
                        self.stats["root_lp"]["fractional_vars_after_cuts"] = self._fractional_list(node)

                    if status == "infeasible":
                        self.nodes_pruned += 1
                        self.pruned_by_infeasibility += 1
                        node_record["status"] = "infeasible_after_cuts"
                        node_record["action"] = "pruned"
                        node_record["pruned_reason"] = "infeasible_after_cuts"
                        self.stats["timeline"].append(node_record)
                        self._log(
                            f"  Nó {self.nodes_explored:4d} (prof={node.depth:2d}) "
                            f"| INVIÁVEL após cortes — podado"
                        )
                        continue

                    if status == "unbounded":
                        node_record["status"] = "unbounded_after_cuts"
                        node_record["action"] = "skipped"
                        self.stats["timeline"].append(node_record)
                        self._log(
                            f"  Nó {self.nodes_explored:4d} (prof={node.depth:2d}) "
                            f"| LP UNBOUNDED após cortes"
                        )
                        continue
        
            self._run_heuristics(node)

            if self._can_prune(node.lp_bound):
                self.nodes_pruned += 1
                self.pruned_by_bound += 1
                node_record["action"] = "pruned"
                node_record["pruned_reason"] = "bound"
                self.stats["timeline"].append(node_record)
                self._log(
                    f"  Nó {self.nodes_explored:4d} (prof={node.depth:2d}) "
                    f"| incumbent={self.best_obj:.4f} "
                    f"| bound={node.lp_bound:.4f} — podado"
                )
                continue

            if node.is_integer_feasible(self._integer_vars):
                self.integer_nodes += 1
                node_record["action"] = "integer"

                if self._is_better(node.lp_bound, self.best_obj):
                    self.best_obj = node.lp_bound
                    self.best_sol = {self._names[i]: node.lp_solution[i] for i in range(self._n)}
                    self.stats["incumbents"].append({
                        "time": self._elapsed_time(),
                        "node": self.nodes_explored,
                        "objective": self.best_obj,
                        "source": "integer_node",
                    })
                    self._log(
                        f"  Nó {self.nodes_explored:4d} (prof={node.depth:2d}) "
                        f"| INTEIRA — novo incumbente: {self.best_obj:.4f}"
                    )

                self.stats["timeline"].append(node_record)
                continue

            t_branch = time.perf_counter()
            frac_var = self.branch_rule.choose_variable(node, self)
            self.branch_rule_calls += 1
            self.branch_rule_time += time.perf_counter() - t_branch

            if frac_var is None:
                frac_var = node.first_fractional(self._integer_vars)
                if frac_var is None:
                    node_record["action"] = "no_fractional_found"
                    self.stats["timeline"].append(node_record)
                    continue

            xval = node.lp_solution[frac_var]
            branch_event = {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "explored_order": self.nodes_explored,
                "depth": node.depth,
                "var_idx": frac_var,
                "var_name": self._names[frac_var],
                "lp_value": xval,
                "fractionality": abs(xval - round(xval)),
                "node_bound": node.lp_bound,
                "rule": repr(self.branch_rule),
            }

            if hasattr(self.branch_rule, "last_candidates"):
                branch_event["candidates"] = self.branch_rule.last_candidates
            if hasattr(self.branch_rule, "last_selected"):
                branch_event["selected"] = self.branch_rule.last_selected

            self.branching_log.append(branch_event)

            node_record["action"] = "branch"
            node_record["branch_var"] = self._names[frac_var]
            node_record["branch_value"] = xval
            self._log(
                f"  Nó {self.nodes_explored:4d} (prof={node.depth:2d}) "
                f"| incumbent={self.best_obj:.4f} "
                f"| bound={node.lp_bound:.4f} "
                f"| branch em {self._names[frac_var]}={node.lp_solution[frac_var]:.3f}"
            )

            child_ceil, child_floor = self._branch(node, frac_var)
            self.strategy.add_children(open_nodes, child_ceil, child_floor, self.direction)
            self.stats["timeline"].append(node_record)

        total_runtime = time.perf_counter() - self.solve_start

        self.stats["tree"] = {
            "nodes_created": self.nodes_created,
            "nodes_explored": self.nodes_explored,
            "nodes_pruned": self.nodes_pruned,
            "pruned_by_infeasibility": self.pruned_by_infeasibility,
            "pruned_by_bound": self.pruned_by_bound,
            "integer_nodes": self.integer_nodes,
            "max_depth": self.max_depth,
            "max_open_nodes": self.max_open_nodes,
            "depth_histogram": dict(self.depth_histogram),
        }

        self.stats["branching"] = {
            "rule": repr(self.branch_rule),
            "calls": self.branch_rule_calls,
            "time": self.branch_rule_time,
            "events": self.branching_log,
        }

        self.stats["heuristics"] = dict(self.heuristic_stats)
        self.stats["cuts"] = dict(self.cut_stats)

        self.stats["final"]["runtime_total"] = total_runtime
        self.stats["final"]["runtime_lp"] = self.lp_time
        self.stats["final"]["runtime_branching"] = self.branch_rule_time
        self.stats["final"]["runtime_heuristics"] = sum(v["time"] for v in self.heuristic_stats.values())
        self.stats["final"]["runtime_cuts"] = sum(v["time"] for v in self.cut_stats.values())
        self.stats["final"]["status"] = "optimal" if self.best_sol is not None else "no_solution_found"
        self.stats["final"]["obj_value"] = self.best_obj if self.best_sol is not None else None

        self._log(f"\n{'='*60}")
        if self.best_sol is not None:
            self._log(f"  Ótimo encontrado  : {self.best_obj:.4f}")
            self._log(f"  Nós explorados    : {self.nodes_explored}")
            self._log(f"  Nós podados       : {self.nodes_pruned}")
            self._log(f"  Chamadas heuríst. : {self.heuristic_calls}")
            self._log(f"  Acertos heuríst.  : {self.heuristic_hits}")
            self._log(f"{'='*60}\n")
            return {
                "status":          "optimal",
                "obj_value":       self.best_obj,
                "solution":        self.best_sol,
                "nodes_explored":  self.nodes_explored,
                "nodes_pruned":    self.nodes_pruned,
                "heuristic_calls": self.heuristic_calls,
                "heuristic_hits":  self.heuristic_hits,
            }
        else:
            self._log("  Nenhuma solução incumbente encontrada.")
            self._log(f"{'='*60}\n")
            return {"status": "no_solution_found"}


    def export_stats(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
