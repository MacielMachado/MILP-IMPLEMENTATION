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

from branch_and_bound import *


class PrimalHeuristic(ABC):
    def __init__(self, probability: float = 1.0):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability deve estar em [0, 1], recebido: {probability}")
        self.probability = probability

    def try_run(self, node: Node, bb) -> Optional[dict]:
        if not self.should_run(node, bb):
            return None
        if random.random() >= self.probability:
            return None
        return self.run(node, bb)

    @abstractmethod
    def should_run(self, node: Node, bb) -> bool:
        pass

    @abstractmethod
    def run(self, node: Node, bb) -> Optional[dict]:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.probability:.0%})"

class RoundingHeuristic(PrimalHeuristic):
    def __init__(self, only_root: bool = False, probability: float = 1.0):
        super().__init__(probability)
        self.only_root = only_root

    def should_run(self, node: Node, bb) -> bool:
        if self.only_root:
            return node.depth == 0
        return True  

    def run(self, node: Node, bb) -> Optional[dict]:
        model = bb._base_model.copy()
        model.setParam("OutputFlag", 0)
        vars_list = model.getVars()

        for i, v in enumerate(vars_list):
            if i in node.lower_bounds:
                v.LB = max(v.LB, node.lower_bounds[i])
            if i in node.upper_bounds:
                v.UB = min(v.UB, node.upper_bounds[i])

        for i in bb._integer_vars:
            rounded = round(node.lp_solution[i])
            rounded = max(rounded, vars_list[i].LB)
            rounded = min(rounded, vars_list[i].UB)
            vars_list[i].LB = rounded
            vars_list[i].UB = rounded

        model.update()
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            return {i: vars_list[i].X for i in range(bb._n)}
        return None

class DivingHeuristic(PrimalHeuristic):
    def __init__(self, only_root: bool = False, max_depth: int = 50, probability: float = 1.0):
        super().__init__(probability)
        self.only_root = only_root
        self.max_depth = max_depth

    def should_run(self, node: Node, bb) -> bool:
        if self.only_root:
            return node.depth == 0
        return not node.is_integer_feasible(bb._integer_vars)

    @abstractmethod
    def select_variable(self, current_solution: dict, integer_vars: set, bb) -> Optional[int]:
        pass

    @abstractmethod
    def choose_value(self, var_idx: int, current_solution: dict, bb) -> float:
        pass

    def run(self, node: Node, bb) -> Optional[dict]:
        dive_lb = dict(node.lower_bounds)
        dive_ub = dict(node.upper_bounds)
        current_solution = dict(node.lp_solution)

        for _ in range(self.max_depth):

            var_idx = self.select_variable(current_solution, bb._integer_vars, bb)

            if var_idx is None:
                return current_solution

            value = self.choose_value(var_idx, current_solution, bb)

            dive_lb[var_idx] = value
            dive_ub[var_idx] = value

            model = bb._base_model.copy()
            model.setParam("OutputFlag", 0)
            vars_list = model.getVars()

            for i, v in enumerate(vars_list):
                if i in dive_lb:
                    v.LB = max(v.LB, dive_lb[i])
                if i in dive_ub:
                    v.UB = min(v.UB, dive_ub[i])

            model.update()
            model.optimize()

            if model.Status != GRB.OPTIMAL:
                return None 
            current_solution = {i: vars_list[i].X for i in range(bb._n)}

        return None  
    
class FractionalDiving(DivingHeuristic):
    def select_variable(self, current_solution, integer_vars, bb) -> Optional[int]:
        best_var  = None
        best_dist = float("inf") 
        for i in integer_vars:
            val  = current_solution.get(i, 0.0)
            frac = abs(val - round(val))
            if frac > 1e-6 and frac < best_dist:
                best_dist = frac
                best_var  = i
        return best_var

    def choose_value(self, var_idx, current_solution, bb) -> float:
        return round(current_solution[var_idx])  

class GuidedDiving(DivingHeuristic):
    def select_variable(self, current_solution, integer_vars, bb) -> Optional[int]:
        if bb.best_sol is None:
            best_var  = None
            best_dist = float("inf")
            for i in integer_vars:
                val  = current_solution.get(i, 0.0)
                frac = abs(val - round(val))
                if frac > 1e-6 and frac < best_dist:
                    best_dist = frac
                    best_var  = i
            return best_var

        best_var  = None
        best_dist = -1.0
        for i in integer_vars:
            val  = current_solution.get(i, 0.0)
            frac = abs(val - round(val))
            if frac <= 1e-6:
                continue
            inc_val  = bb.best_sol.get(bb._names[i], 0.0)
            distance = abs(val - inc_val)
            if distance > best_dist:
                best_dist = distance
                best_var  = i
        return best_var

    def choose_value(self, var_idx, current_solution, bb) -> float:
        if bb.best_sol is None:
            return round(current_solution[var_idx])
        inc_val = bb.best_sol.get(bb._names[var_idx], 0.0)
        return round(inc_val)

class CoefficientDiving(DivingHeuristic):
    def select_variable(self, current_solution, integer_vars, bb) -> Optional[int]:
        obj_coeffs = [v.Obj for v in bb._base_model.getVars()]
        sense      = bb._sense  

        best_var    = None
        best_score  = -float("inf")
        for i in integer_vars:
            val  = current_solution.get(i, 0.0)
            frac = abs(val - round(val))
            if frac <= 1e-6:
                continue
            score = abs(obj_coeffs[i]) if sense == GRB.MINIMIZE else obj_coeffs[i]
            if score > best_score:
                best_score = score
                best_var   = i
        return best_var

    def choose_value(self, var_idx, current_solution, bb) -> float:
        obj_val = bb._base_model.getVars()[var_idx].Obj
        if bb._sense == GRB.MAXIMIZE:
            return math.ceil(current_solution[var_idx])
        else:
            return math.floor(current_solution[var_idx])

class LockDiving(DivingHeuristic):
    def __init__(self, only_root: bool = False, max_depth: int = 50, probability: float = 1.0):
        super().__init__(only_root, max_depth, probability)
        self._lock_cache = None  

    def _compute_locks(self, bb) -> dict:
        locks     = {i: [0, 0] for i in bb._integer_vars}
        model     = bb._base_model
        constrs   = model.getConstrs()
        vars_list = model.getVars()

        var_id_to_idx = {id(v): i for i, v in enumerate(vars_list)}

        for c in constrs:
            sense = c.Sense
            row   = model.getRow(c)
            for k in range(row.size()):
                v   = row.getVar(k)
                idx = var_id_to_idx.get(id(v), -1)
                if idx not in locks:
                    continue
                coef = row.getCoeff(k)
                if sense == '<':
                    if coef > 0:
                        locks[idx][0] += 1
                    else:
                        locks[idx][1] += 1
                elif sense == '>':
                    if coef < 0:
                        locks[idx][0] += 1
                    else:
                        locks[idx][1] += 1

        return {i: tuple(v) for i, v in locks.items()}

    def select_variable(self, current_solution, integer_vars, bb) -> Optional[int]:
        if self._lock_cache is None:
            self._lock_cache = self._compute_locks(bb)

        best_var   = None
        best_score = float("inf")
        for i in integer_vars:
            val  = current_solution.get(i, 0.0)
            frac = abs(val - round(val))
            if frac <= 1e-6:
                continue
            locks_up, locks_down = self._lock_cache.get(i, (0, 0))
            score = min(locks_up, locks_down)
            if score < best_score:
                best_score = score
                best_var   = i
        return best_var

    def choose_value(self, var_idx, current_solution, bb) -> float:
        if self._lock_cache is None:
            self._lock_cache = self._compute_locks(bb)
        locks_up, locks_down = self._lock_cache.get(var_idx, (0, 0))
        val = current_solution[var_idx]
        if locks_up <= locks_down:
            return math.ceil(val)
        else:
            return math.floor(val)

class FeasibilityPump(PrimalHeuristic):
    def __init__(self, only_root: bool = True, max_iter: int = 50, probability: float = 1.0):
        super().__init__(probability)
        self.only_root = only_root
        self.max_iter  = max_iter

    def should_run(self, node: Node, bb) -> bool:
        if self.only_root:
            return node.depth == 0 and bb.best_sol is None
        return bb.best_sol is None

    def run(self, node: Node, bb) -> Optional[dict]:
        current_lp = dict(node.lp_solution)

        for _ in range(self.max_iter):
            x_bar = {
                i: round(current_lp[i])
                for i in bb._integer_vars
                if i in current_lp
            }

            model = bb._base_model.copy()
            model.setParam("OutputFlag", 0)
            vars_list = model.getVars()

            for i, v in enumerate(vars_list):
                if i in node.lower_bounds:
                    v.LB = max(v.LB, node.lower_bounds[i])
                if i in node.upper_bounds:
                    v.UB = min(v.UB, node.upper_bounds[i])

            t = {}
            for i in bb._integer_vars:
                if i not in x_bar:
                    continue
                t[i] = model.addVar(lb=0.0, name=f"_pump_t_{i}")
                model.addConstr(t[i] >= vars_list[i] - x_bar[i])
                model.addConstr(t[i] >= x_bar[i] - vars_list[i])

            model.setObjective(gp.quicksum(t[i] for i in t), GRB.MINIMIZE)
            model.update()
            model.optimize()

            if model.Status != GRB.OPTIMAL:
                return None

            current_lp = {i: vars_list[i].X for i in range(bb._n)}

            if all(
                abs(current_lp[i] - round(current_lp[i])) <= 1e-6
                for i in bb._integer_vars
                if i in current_lp
            ):
                return current_lp

        return None

class FeasibilityJump(PrimalHeuristic):
    def __init__(self, only_root: bool = True, max_iter: int = 100, max_restarts: int = 5, probability: float = 1.0):
        super().__init__(probability)
        self.only_root    = only_root
        self.max_iter     = max_iter
        self.max_restarts = max_restarts

    def should_run(self, node: Node, bb) -> bool:
        if self.only_root:
            return node.depth == 0 and bb.best_sol is None
        return bb.best_sol is None

    def run(self, node: Node, bb) -> Optional[dict]:
        current_lp       = dict(node.lp_solution)
        seen_projections = set()
        restarts         = 0

        for _ in range(self.max_iter):
            x_bar = {
                i: round(current_lp[i])
                for i in bb._integer_vars
                if i in current_lp
            }

            proj_key = tuple(sorted(x_bar.items()))
            if proj_key in seen_projections:
                if restarts >= self.max_restarts:
                    return None
                restarts += 1
                int_vars_list = list(x_bar.keys())
                n_perturb = max(1, len(int_vars_list) // 5)
                for i in random.sample(int_vars_list, n_perturb):
                    val = current_lp[i]
                    x_bar[i] = math.ceil(val) if x_bar[i] == math.floor(val) else math.floor(val)

            seen_projections.add(proj_key)

            model = bb._base_model.copy()
            model.setParam("OutputFlag", 0)
            vars_list = model.getVars()

            for i, v in enumerate(vars_list):
                if i in node.lower_bounds:
                    v.LB = max(v.LB, node.lower_bounds[i])
                if i in node.upper_bounds:
                    v.UB = min(v.UB, node.upper_bounds[i])

            t = {}
            for i in bb._integer_vars:
                if i not in x_bar:
                    continue
                t[i] = model.addVar(lb=0.0, name=f"_jump_t_{i}")
                model.addConstr(t[i] >= vars_list[i] - x_bar[i])
                model.addConstr(t[i] >= x_bar[i] - vars_list[i])

            model.setObjective(gp.quicksum(t[i] for i in t), GRB.MINIMIZE)
            model.update()
            model.optimize()

            if model.Status != GRB.OPTIMAL:
                return None

            current_lp = {i: vars_list[i].X for i in range(bb._n)}

            if all(
                abs(current_lp[i] - round(current_lp[i])) <= 1e-6
                for i in bb._integer_vars
                if i in current_lp
            ):
                return current_lp

        return None

class FixAndPropagate(PrimalHeuristic):
    def __init__(self, only_root: bool = False, max_depth: int = 10000, probability: float = 1.0):
        super().__init__(probability)
        self.only_root = only_root
        self.max_depth = max_depth

    def should_run(self, node: Node, bb) -> bool:
        if self.only_root:
            return node.depth == 0
        return True  

    def run(self, node: Node, bb) -> Optional[dict]:
        fix_lb = dict(node.lower_bounds)
        fix_ub = dict(node.upper_bounds)
        current_solution = dict(node.lp_solution)

        for _ in range(self.max_depth):

            fractional = [
                (i, abs(current_solution[i] - round(current_solution[i])))
                for i in bb._integer_vars
                if i in current_solution
                and abs(current_solution[i] - round(current_solution[i])) > 1e-6
            ]

            if not fractional:
                return current_solution

            fractional.sort(key=lambda x: x[1])
            var_idx, _ = fractional[0]

            value = round(current_solution[var_idx])
            value = max(value, fix_lb.get(var_idx, bb._orig_lb[var_idx]))
            value = min(value, fix_ub.get(var_idx, bb._orig_ub[var_idx]))
            fix_lb[var_idx] = value
            fix_ub[var_idx] = value

            model = bb._base_model.copy()
            model.setParam("OutputFlag", 0)
            vars_list = model.getVars()

            for i, v in enumerate(vars_list):
                if i in fix_lb:
                    v.LB = max(v.LB, fix_lb[i])
                if i in fix_ub:
                    v.UB = min(v.UB, fix_ub[i])

            model.update()
            model.optimize()

            if model.Status != GRB.OPTIMAL:
                return None  
            current_solution = {i: vars_list[i].X for i in range(bb._n)}

            for i in bb._integer_vars:
                v = vars_list[i]
                if abs(v.LB - v.UB) < 1e-6:
                    fix_lb[i] = v.LB
                    fix_ub[i] = v.UB

        return None

class RENS(PrimalHeuristic):
    def __init__(
        self,
        only_root: bool = True,
        fix_ratio: float = 0.0,
        adaptive_ratios: list = None,
        time_limit: float = 10.0,
        probability: float = 1.0,
    ):
        super().__init__(probability)
        self.only_root       = only_root
        self.fix_ratio       = fix_ratio
        self.adaptive_ratios = adaptive_ratios if adaptive_ratios is not None else [0.3, 0.5, 0.7]
        self.time_limit      = time_limit

    def should_run(self, node: Node, bb) -> bool:
        if self.only_root:
            return node.depth == 0
        return True

    def run(self, node: Node, bb) -> Optional[dict]:
        lp_sol       = node.lp_solution
        integer_vars = bb._integer_vars
        tol          = 1e-6

        fixed_vars     = {}  
        fractional_vars = []

        for i in integer_vars:
            val  = lp_sol.get(i, 0.0)
            frac = abs(val - round(val))
            if frac <= tol:
                fixed_vars[i] = round(val)
            else:
                fractional_vars.append((i, frac))

        fractional_vars.sort(key=lambda x: x[1])

        ratios_to_try = [self.fix_ratio] + self.adaptive_ratios

        for ratio in ratios_to_try:
            solution = self._solve_submip(
                node, bb, fixed_vars, fractional_vars, ratio
            )
            if solution is not None:
                return solution

        return None

    def _solve_submip(
        self,
        node: Node,
        bb,
        fixed_vars: dict,
        fractional_vars: list,
        extra_fix_ratio: float,
    ) -> Optional[dict]:
        model = bb._base_model.copy()
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.time_limit)
        vars_list = model.getVars()

        for i, vtype in enumerate(bb._orig_vtype):
            if vtype in (GRB.BINARY, GRB.INTEGER):
                vars_list[i].VType = vtype

        for i, v in enumerate(vars_list):
            if i in node.lower_bounds:
                v.LB = max(v.LB, node.lower_bounds[i])
            if i in node.upper_bounds:
                v.UB = min(v.UB, node.upper_bounds[i])

        for i, val in fixed_vars.items():
            vars_list[i].LB = val
            vars_list[i].UB = val

        for i, frac in fractional_vars:
            val = bb._base_model.getVars()[i].LB  
            lp_val = node.lp_solution.get(i, 0.0)
            vars_list[i].LB = max(vars_list[i].LB, math.floor(lp_val))
            vars_list[i].UB = min(vars_list[i].UB, math.ceil(lp_val))

        n_extra = int(len(fractional_vars) * extra_fix_ratio)
        for i, frac in fractional_vars[:n_extra]:
            lp_val = node.lp_solution.get(i, 0.0)
            rounded = round(lp_val)
            vars_list[i].LB = rounded
            vars_list[i].UB = rounded

        model.update()
        model.optimize()

        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            if model.SolCount > 0:
                return {i: vars_list[i].X for i in range(bb._n)}

        return None

class RINS(PrimalHeuristic):
    def __init__(
        self,
        only_root: bool = False,
        tau: float = 0.5,
        fix_ratio: float = 0.0,
        adaptive_ratios: list = None,
        time_limit: float = 10.0,
        probability: float = 1.0,
    ):
        super().__init__(probability)
        self.only_root       = only_root
        self.tau             = tau
        self.fix_ratio       = fix_ratio
        self.adaptive_ratios = adaptive_ratios if adaptive_ratios is not None else [0.3, 0.5, 0.7]
        self.time_limit      = time_limit

    def should_run(self, node: Node, bb) -> bool:
        if bb.best_sol is None:
            return False
        if self.only_root:
            return node.depth == 0
        return True

    def run(self, node: Node, bb) -> Optional[dict]:
        lp_sol       = node.lp_solution
        integer_vars = bb._integer_vars

        fixed_vars = {}  
        free_vars  = []  

        rc = self._get_reduced_costs(node, bb)

        for i in integer_vars:
            lp_val  = lp_sol.get(i, 0.0)
            inc_val = round(bb.best_sol.get(bb._names[i], 0.0))

            if abs(lp_val - inc_val) <= self.tau:
                fixed_vars[i] = inc_val
            else:
                disagreement = abs(lp_val - inc_val)
                reduced_cost = abs(rc.get(i, 0.0))
                epsilon      = 1e-8
                score = disagreement / (reduced_cost + epsilon)
                free_vars.append((i, score))

        free_vars.sort(key=lambda x: x[1])

        ratios_to_try = [self.fix_ratio] + self.adaptive_ratios

        for ratio in ratios_to_try:
            solution = self._solve_submip(node, bb, fixed_vars, free_vars, ratio)
            if solution is not None:
                return solution

        return None

    def _get_reduced_costs(self, node: Node, bb) -> dict:
        model = bb._base_model.copy()
        model.setParam("OutputFlag", 0)
        vars_list = model.getVars()

        for i, v in enumerate(vars_list):
            if i in node.lower_bounds:
                v.LB = max(v.LB, node.lower_bounds[i])
            if i in node.upper_bounds:
                v.UB = min(v.UB, node.upper_bounds[i])

        model.update()
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            return {i: vars_list[i].RC for i in range(bb._n)}
        return {}

    def _solve_submip(
        self,
        node: Node,
        bb,
        fixed_vars: dict,
        free_vars: list,
        extra_fix_ratio: float,
    ) -> Optional[dict]:
        model = bb._base_model.copy()
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.time_limit)
        vars_list = model.getVars()

        for i, vtype in enumerate(bb._orig_vtype):
            if vtype in (GRB.BINARY, GRB.INTEGER):
                vars_list[i].VType = vtype

        for i, v in enumerate(vars_list):
            if i in node.lower_bounds:
                v.LB = max(v.LB, node.lower_bounds[i])
            if i in node.upper_bounds:
                v.UB = min(v.UB, node.upper_bounds[i])

        for i, val in fixed_vars.items():
            val = max(val, vars_list[i].LB)
            val = min(val, vars_list[i].UB)
            vars_list[i].LB = val
            vars_list[i].UB = val

        n_extra = int(len(free_vars) * extra_fix_ratio)
        for i, score in free_vars[:n_extra]:
            inc_val = round(bb.best_sol.get(bb._names[i], 0.0))
            inc_val = max(inc_val, vars_list[i].LB)
            inc_val = min(inc_val, vars_list[i].UB)
            vars_list[i].LB = inc_val
            vars_list[i].UB = inc_val

        model.update()
        model.optimize()

        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            if model.SolCount > 0:
                obj = model.ObjVal
                if bb._is_better(obj, bb.best_obj):
                    return {i: vars_list[i].X for i in range(bb._n)}

        return None

class LocalSearch(PrimalHeuristic):
    def __init__(
        self,
        only_root: bool = False,
        window_size: int = 15,
        max_iter: int = 10,
        beta: float = 0.1,
        time_limit: float = 10.0,
        probability: float = 1.0,
    ):
        super().__init__(probability)
        self.only_root   = only_root
        self.window_size = window_size
        self.max_iter    = max_iter
        self.beta        = beta
        self.time_limit  = time_limit

    def should_run(self, node: Node, bb) -> bool:
        if bb.best_sol is None:
            return False
        if self.only_root:
            return node.depth == 0
        return True

    def run(self, node: Node, bb) -> Optional[dict]:
        lp_sol       = node.lp_solution
        integer_vars = bb._integer_vars

        rc = self._get_reduced_costs(node, bb)

        scored = []
        for i in integer_vars:
            lp_val  = lp_sol.get(i, 0.0)
            inc_val = round(bb.best_sol.get(bb._names[i], 0.0))
            disagreement = abs(lp_val - inc_val)
            reduced_cost = abs(rc.get(i, 0.0))
            score = disagreement + self.beta * reduced_cost
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_vars = [i for i, _ in scored]
        n_vars      = len(ranked_vars)
        w           = min(self.window_size, n_vars)

        best_solution = None

        for iteration in range(min(self.max_iter, n_vars)):
            window = {ranked_vars[(iteration + k) % n_vars] for k in range(w)}

            solution = self._solve_window(node, bb, window)
            if solution is not None:
                best_solution = solution

        return best_solution

    def _get_reduced_costs(self, node: Node, bb) -> dict:
        model = bb._base_model.copy()
        model.setParam("OutputFlag", 0)
        vars_list = model.getVars()

        for i, v in enumerate(vars_list):
            if i in node.lower_bounds:
                v.LB = max(v.LB, node.lower_bounds[i])
            if i in node.upper_bounds:
                v.UB = min(v.UB, node.upper_bounds[i])

        model.update()
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            return {i: vars_list[i].RC for i in range(bb._n)}
        return {}

    def _solve_window(self, node: Node, bb, window: set) -> Optional[dict]:
        model = bb._base_model.copy()
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.time_limit)
        vars_list = model.getVars()

        for i, vtype in enumerate(bb._orig_vtype):
            if vtype in (GRB.BINARY, GRB.INTEGER):
                vars_list[i].VType = vtype

        for i, v in enumerate(vars_list):
            if i in node.lower_bounds:
                v.LB = max(v.LB, node.lower_bounds[i])
            if i in node.upper_bounds:
                v.UB = min(v.UB, node.upper_bounds[i])

        for i in bb._integer_vars:
            if i not in window:
                inc_val = round(bb.best_sol.get(bb._names[i], 0.0))
                inc_val = max(inc_val, vars_list[i].LB)
                inc_val = min(inc_val, vars_list[i].UB)
                vars_list[i].LB = inc_val
                vars_list[i].UB = inc_val

        model.update()
        model.optimize()

        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            if model.SolCount > 0:
                obj = model.ObjVal
                if bb._is_better(obj, bb.best_obj):
                    return {i: vars_list[i].X for i in range(bb._n)}

        return None

class LocalBranching(PrimalHeuristic):
    def __init__(
        self,
        only_root: bool = False,
        k: int = 20,
        radius_ratio: float = 0.1,
        add_obj_cut: bool = True,
        time_limit: float = 15.0,
        probability: float = 1.0,
    ):
        super().__init__(probability)
        self.only_root    = only_root
        self.k            = k
        self.radius_ratio = radius_ratio
        self.add_obj_cut  = add_obj_cut
        self.time_limit   = time_limit

    def should_run(self, node: Node, bb) -> bool:
        if bb.best_sol is None:
            return False
        has_binary = any(
            bb._orig_vtype[i] == GRB.BINARY
            for i in bb._integer_vars
        )
        if not has_binary:
            return False
        if self.only_root:
            return node.depth == 0
        return True

    def run(self, node: Node, bb) -> Optional[dict]:
        lp_sol       = node.lp_solution
        integer_vars = bb._integer_vars

        binary_vars = [
            i for i in integer_vars
            if bb._orig_vtype[i] == GRB.BINARY
        ]
        general_int_vars = [
            i for i in integer_vars
            if bb._orig_vtype[i] == GRB.INTEGER
        ]

        if not binary_vars:
            return None

        k = self.k if self.k is not None else max(
            1, round(self.radius_ratio * len(binary_vars))
        )

        model = bb._base_model.copy()
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.time_limit)
        vars_list = model.getVars()

        for i, vtype in enumerate(bb._orig_vtype):
            if vtype in (GRB.BINARY, GRB.INTEGER):
                vars_list[i].VType = vtype

        for i, v in enumerate(vars_list):
            if i in node.lower_bounds:
                v.LB = max(v.LB, node.lower_bounds[i])
            if i in node.upper_bounds:
                v.UB = min(v.UB, node.upper_bounds[i])

        tol = 0.5
        for i in general_int_vars:
            lp_val  = lp_sol.get(i, 0.0)
            inc_val = round(bb.best_sol.get(bb._names[i], 0.0))
            if abs(lp_val - inc_val) <= tol:
                inc_val = max(inc_val, vars_list[i].LB)
                inc_val = min(inc_val, vars_list[i].UB)
                vars_list[i].LB = inc_val
                vars_list[i].UB = inc_val

        inc_vals = {
            i: round(bb.best_sol.get(bb._names[i], 0.0))
            for i in binary_vars
        }

        hamming_expr = gp.LinExpr()
        for i in binary_vars:
            if inc_vals[i] == 1:
                hamming_expr += (1.0 - vars_list[i])
            else:
                hamming_expr += vars_list[i]

        model.addConstr(hamming_expr <= k, name="_lb_hamming")

      
        if self.add_obj_cut:
            epsilon = 1e-4
            obj_expr = gp.quicksum(
                bb._base_model.getVars()[i].Obj * vars_list[i]
                for i in range(bb._n)
            )
            if bb._sense == GRB.MAXIMIZE:
                model.addConstr(obj_expr >= bb.best_obj + epsilon, name="_lb_improve")
            else:
                model.addConstr(obj_expr <= bb.best_obj - epsilon, name="_lb_improve")

        model.update()
        model.optimize()

        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            if model.SolCount > 0:
                obj = model.ObjVal
                if bb._is_better(obj, bb.best_obj):
                    return {i: vars_list[i].X for i in range(bb._n)}

        return None

class LNSDestroyRule(ABC):
    @abstractmethod
    def choose_free_set(self, node: Node, bb, destroy_rate: float) -> set:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


class RandomDestroy(LNSDestroyRule):
    def choose_free_set(self, node: Node, bb, destroy_rate: float) -> set:
        int_vars = list(bb._integer_vars)
        k = max(1, int(destroy_rate * len(int_vars)))
        return set(random.sample(int_vars, min(k, len(int_vars))))


class FractionalGuidedDestroy(LNSDestroyRule):
    def choose_free_set(self, node: Node, bb, destroy_rate: float) -> set:
        if bb.best_sol is None:
            return RandomDestroy().choose_free_set(node, bb, destroy_rate)

        int_vars = list(bb._integer_vars)
        k = max(1, int(destroy_rate * len(int_vars)))

        scored = []
        for i in int_vars:
            lp_val  = node.lp_solution.get(i, 0.0)
            inc_val = round(bb.best_sol.get(bb._names[i], 0.0))
            scored.append((i, abs(lp_val - inc_val)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return {i for i, _ in scored[:k]}


class ConstraintBasedDestroy(LNSDestroyRule):
    def choose_free_set(self, node: Node, bb, destroy_rate: float) -> set:
        if bb.best_sol is None:
            return RandomDestroy().choose_free_set(node, bb, destroy_rate)

        int_vars   = list(bb._integer_vars)
        k          = max(1, int(destroy_rate * len(int_vars)))
        model      = bb._base_model
        constrs    = model.getConstrs()
        vars_list  = model.getVars()
        var_id_idx = {id(v): i for i, v in enumerate(vars_list)}

        tightness = {}  
        for c_idx, c in enumerate(constrs):
            row = model.getRow(c)
            lhs = sum(
                row.getCoeff(k2) * bb.best_sol.get(bb._names[var_id_idx[id(row.getVar(k2))]], 0.0)
                for k2 in range(row.size())
                if id(row.getVar(k2)) in var_id_idx
            )
            rhs   = c.RHS
            sense = c.Sense
            if sense == '<':
                slack = rhs - lhs
            elif sense == '>':
                slack = lhs - rhs
            else:
                slack = abs(lhs - rhs)
            tightness[c_idx] = slack

        tight_constrs = sorted(tightness.keys(), key=lambda x: tightness[x])
        n_tight = max(1, len(tight_constrs) // 4)  

        candidate_vars = set()
        for c_idx in tight_constrs[:n_tight]:
            row = model.getRow(constrs[c_idx])
            for k2 in range(row.size()):
                v   = row.getVar(k2)
                idx = var_id_idx.get(id(v), -1)
                if idx in bb._integer_vars:
                    candidate_vars.add(idx)

        if len(candidate_vars) < k:
            remaining = [i for i in int_vars if i not in candidate_vars]
            if remaining:
                extra = random.sample(remaining, min(k - len(candidate_vars), len(remaining)))
                candidate_vars.update(extra)

        candidates = list(candidate_vars)
        return set(candidates[:k])


class ObjectiveBasedDestroy(LNSDestroyRule):
    def choose_free_set(self, node: Node, bb, destroy_rate: float) -> set:
        int_vars = list(bb._integer_vars)
        k = max(1, int(destroy_rate * len(int_vars)))

        lp_model  = bb._base_model.copy()
        lp_model.setParam("OutputFlag", 0)
        lp_vars   = lp_model.getVars()

        for i, v in enumerate(lp_vars):
            if i in node.lower_bounds:
                v.LB = max(v.LB, node.lower_bounds[i])
            if i in node.upper_bounds:
                v.UB = min(v.UB, node.upper_bounds[i])

        lp_model.update()
        lp_model.optimize()

        if lp_model.Status == GRB.OPTIMAL:
            rc = {i: abs(lp_vars[i].RC) for i in int_vars}
        else:
            rc = {i: abs(bb._base_model.getVars()[i].Obj) for i in int_vars}

        scored = sorted(int_vars, key=lambda i: rc.get(i, 0.0), reverse=True)
        return set(scored[:k])


class LNS(PrimalHeuristic):
    def __init__(
        self,
        destroy_rule: LNSDestroyRule = None,
        only_root: bool = False,
        destroy_rate: float = 0.3,
        max_destroy_rate: float = 0.7,
        max_iter: int = 5,
        patience: int = 2,
        time_limit: float = 10.0,
        probability: float = 1.0,
    ):
        super().__init__(probability)
        self.destroy_rule     = destroy_rule or FractionalGuidedDestroy()
        self.only_root        = only_root
        self.destroy_rate     = destroy_rate
        self.max_destroy_rate = max_destroy_rate
        self.max_iter         = max_iter
        self.patience         = patience
        self.time_limit       = time_limit

    def should_run(self, node: Node, bb) -> bool:
        if bb.best_sol is None:
            return False
        if self.only_root:
            return node.depth == 0
        return True

    def __repr__(self) -> str:
        return f"LNS({self.destroy_rule}, p={self.probability:.0%})"

    def run(self, node: Node, bb) -> Optional[dict]:
        best_solution = None
        current_rate  = self.destroy_rate
        no_improve    = 0

        for _ in range(self.max_iter):
            free_set = self.destroy_rule.choose_free_set(node, bb, current_rate)

            solution = self._repair(node, bb, free_set)

            if solution is not None:
                obj = sum(
                    bb._base_model.getVars()[i].Obj * solution[i]
                    for i in range(bb._n)
                )
                if bb._is_better(obj, bb.best_obj):
                    best_solution = solution
                    no_improve    = 0
                    current_rate = max(self.destroy_rate, current_rate * 0.9)
                    continue

            no_improve += 1
            if no_improve >= self.patience:
                current_rate = min(self.max_destroy_rate, current_rate * 1.2)
                no_improve   = 0

        return best_solution

    def _repair(self, node: Node, bb, free_set: set) -> Optional[dict]:
        model = bb._base_model.copy()
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.time_limit)
        vars_list = model.getVars()

        for i, vtype in enumerate(bb._orig_vtype):
            if vtype in (GRB.BINARY, GRB.INTEGER):
                vars_list[i].VType = vtype

        for i, v in enumerate(vars_list):
            if i in node.lower_bounds:
                v.LB = max(v.LB, node.lower_bounds[i])
            if i in node.upper_bounds:
                v.UB = min(v.UB, node.upper_bounds[i])

        for i in bb._integer_vars:
            if i not in free_set:
                inc_val = round(bb.best_sol.get(bb._names[i], 0.0))
                inc_val = max(inc_val, vars_list[i].LB)
                inc_val = min(inc_val, vars_list[i].UB)
                vars_list[i].LB = inc_val
                vars_list[i].UB = inc_val

        model.update()
        model.optimize()

        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            if model.SolCount > 0:
                obj = model.ObjVal
                if bb._is_better(obj, bb.best_obj):
                    return {i: vars_list[i].X for i in range(bb._n)}

        return None


class ALNSAcceptance(Enum):
    """Critério de aceitação do ALNS."""
    GREEDY           = "greedy"            
    SIMULATED_ANNEAL = "simulated_anneal"  
    THRESHOLD        = "threshold"         


class ALNS(PrimalHeuristic):
    def __init__(
        self,
        destroy_rules: list = None,
        only_root: bool = False,
        destroy_rate: float = 0.3,
        max_iter: int = 10,
        eta: float = 0.2,
        w_min: float = 0.01,
        sigma_1: float = 33.0,
        sigma_2: float = 9.0,
        sigma_3: float = 3.0,
        segment_size: int = 5,
        acceptance: ALNSAcceptance = ALNSAcceptance.GREEDY,
        temp_init: float = None,
        temp_decay: float = 0.995,
        threshold_delta: float = 0.05,
        time_limit: float = 10.0,
        probability: float = 1.0,
    ):
        super().__init__(probability)
        self.destroy_rules    = destroy_rules or [
            RandomDestroy(),
            FractionalGuidedDestroy(),
            ConstraintBasedDestroy(),
            ObjectiveBasedDestroy(),
        ]
        self.only_root        = only_root
        self.destroy_rate     = destroy_rate
        self.max_iter         = max_iter
        self.eta              = eta
        self.w_min            = w_min
        self.sigma_1          = sigma_1
        self.sigma_2          = sigma_2
        self.sigma_3          = sigma_3
        self.segment_size     = segment_size
        self.acceptance       = acceptance
        self.temp_init        = temp_init
        self.temp_decay       = temp_decay
        self.threshold_delta  = threshold_delta
        self.time_limit       = time_limit

        self._weights = [1.0] * len(self.destroy_rules)

        self._seg_scores = [0.0] * len(self.destroy_rules)
        self._seg_counts = [0]   * len(self.destroy_rules)
        self._seg_iter   = 0

        self._temperature = None

        self.weight_history = []

    def should_run(self, node: Node, bb) -> bool:
        if bb.best_sol is None:
            return False
        if self.only_root:
            return node.depth == 0
        return True

    def __repr__(self) -> str:
        ops = [repr(r) for r in self.destroy_rules]
        return f"ALNS({ops}, p={self.probability:.0%})"

    def _roulette_select(self) -> int:
        total = sum(self._weights)
        u     = random.random() * total
        cumul = 0.0
        for idx, w in enumerate(self._weights):
            cumul += w
            if cumul >= u:
                return idx
        return len(self._weights) - 1

    def _update_weight(self, op_idx: int, sigma: float) -> None:
        self._seg_scores[op_idx] += sigma
        self._seg_counts[op_idx] += 1
        self._seg_iter            += 1

        if self._seg_iter >= self.segment_size:
            for k in range(len(self._weights)):
                if self._seg_counts[k] > 0:
                    avg_sigma = self._seg_scores[k] / self._seg_counts[k]
                    self._weights[k] = max(
                        self.w_min,
                        (1 - self.eta) * self._weights[k] + self.eta * avg_sigma
                    )
            self.weight_history.append(list(self._weights))
            self._seg_scores = [0.0] * len(self.destroy_rules)
            self._seg_counts = [0]   * len(self.destroy_rules)
            self._seg_iter   = 0

    def _initialize_temperature(self, bb, candidate_obj: float) -> None:
        if self.temp_init is not None:
            self._temperature = self.temp_init
            return
        if bb.best_obj not in (float("inf"), -float("inf")):
            delta = abs(bb.best_obj * 0.05) 
            if delta > 1e-8:
                self._temperature = -delta / math.log(0.5)
                return
        self._temperature = abs(candidate_obj) * 0.05 / math.log(2) + 1e-8

    def _accept(self, candidate_obj: float, current_obj: float, bb) -> bool:
        if self.acceptance == ALNSAcceptance.GREEDY:
            return bb._is_better(candidate_obj, current_obj)

        elif self.acceptance == ALNSAcceptance.SIMULATED_ANNEAL:
            if bb._is_better(candidate_obj, current_obj):
                return True
            if self._temperature is None or self._temperature < 1e-10:
                return False
            if bb._sense == GRB.MINIMIZE:
                delta = candidate_obj - current_obj
            else:
                delta = current_obj - candidate_obj
            prob = math.exp(-delta / self._temperature)
            return random.random() < prob

        elif self.acceptance == ALNSAcceptance.THRESHOLD:
            if bb._sense == GRB.MINIMIZE:
                return candidate_obj <= current_obj * (1 + self.threshold_delta)
            else:
                return candidate_obj >= current_obj * (1 - self.threshold_delta)

        return False

    def _repair(self, node: Node, bb, free_set: set) -> Optional[tuple]:
        model = bb._base_model.copy()
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", self.time_limit)
        vars_list = model.getVars()

        for i, vtype in enumerate(bb._orig_vtype):
            if vtype in (GRB.BINARY, GRB.INTEGER):
                vars_list[i].VType = vtype

        for i, v in enumerate(vars_list):
            if i in node.lower_bounds:
                v.LB = max(v.LB, node.lower_bounds[i])
            if i in node.upper_bounds:
                v.UB = min(v.UB, node.upper_bounds[i])

        for i in bb._integer_vars:
            if i not in free_set:
                inc_val = round(bb.best_sol.get(bb._names[i], 0.0))
                inc_val = max(inc_val, vars_list[i].LB)
                inc_val = min(inc_val, vars_list[i].UB)
                vars_list[i].LB = inc_val
                vars_list[i].UB = inc_val

        model.update()
        model.optimize()

        if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
            sol = {i: vars_list[i].X for i in range(bb._n)}
            obj = model.ObjVal
            return sol, obj

        return None

    def run(self, node: Node, bb) -> Optional[dict]:
        current_obj  = bb.best_obj
        global_best  = bb.best_obj
        best_solution = None

        for t in range(self.max_iter):
            op_idx = self._roulette_select()
            rule   = self.destroy_rules[op_idx]

            free_set = rule.choose_free_set(node, bb, self.destroy_rate)

            result = self._repair(node, bb, free_set)
            if result is None:
                self._update_weight(op_idx, 0.0)  
                continue

            cand_sol, cand_obj = result

            if t == 0 and self._temperature is None:
                self._initialize_temperature(bb, cand_obj)

            sigma = 0.0

            if bb._is_better(cand_obj, global_best):
                global_best   = cand_obj
                current_obj   = cand_obj
                best_solution = cand_sol
                sigma         = self.sigma_1

            elif self._accept(cand_obj, current_obj, bb):
                if bb._is_better(cand_obj, current_obj):
                    sigma = self.sigma_2 
                else:
                    sigma = self.sigma_3 
                current_obj = cand_obj
                if best_solution is None:
                    best_solution = cand_sol

            else:
                sigma = 0.0  

            self._update_weight(op_idx, sigma)

            if self._temperature is not None:
                self._temperature *= self.temp_decay

        return best_solution

