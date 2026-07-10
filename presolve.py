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


class PresolveResult:
    def __init__(self):
        self.infeasible = False
        self.bounds_tightened = 0
        self.rows_removed = 0
        self.vars_fixed = 0
        self.coefficients_changed = 0
        self.runtime = 0.0

    def to_dict(self):
        return {
            "infeasible": self.infeasible,
            "bounds_tightened": self.bounds_tightened,
            "rows_removed": self.rows_removed,
            "vars_fixed": self.vars_fixed,
            "coefficients_changed": self.coefficients_changed,
            "runtime": self.runtime,
        }

    def __repr__(self) -> str:
        if self.infeasible:
            return "PresolveResult(INFEASIBLE)"
        return (
            "PresolveResult("
            f"bounds_tightened={self.bounds_tightened}, "
            f"rows_removed={self.rows_removed}, "
            f"vars_fixed={self.vars_fixed}, "
            f"coefficients_changed={self.coefficients_changed})"
        )

class PresolveRule(ABC):
    @abstractmethod
    def apply(self, model: gp.Model) -> PresolveResult:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


class BoundTightening(PresolveRule):
    def __init__(self, max_rounds: int = 10, eps: float = 1e-8, verbose: bool = False):
        self.max_rounds = max_rounds
        self.eps        = eps
        self.verbose    = verbose

    def apply(self, model: gp.Model) -> PresolveResult:
        result    = PresolveResult()
        vars_list = model.getVars()
        constrs   = model.getConstrs()

        var_id_to_idx = {id(v): i for i, v in enumerate(vars_list)}

        lb = [v.LB for v in vars_list]
        ub = [v.UB for v in vars_list]

        vtype = [v.VType for v in vars_list]

        rows = []
        for c in constrs:
            row_gurobi = model.getRow(c)
            coeffs     = []
            for k in range(row_gurobi.size()):
                v   = row_gurobi.getVar(k)
                idx = var_id_to_idx.get(id(v), -1)
                if idx >= 0:
                    coeffs.append((idx, row_gurobi.getCoeff(k)))
            rows.append({
                "coeffs": coeffs,
                "lhs":    c.RHS if c.Sense == ">" else -math.inf,
                "rhs":    c.RHS if c.Sense == "<" else (
                              c.RHS if c.Sense == "=" else math.inf
                          ),
                "sense":  c.Sense,
            })

        for r in rows:
            if r["sense"] == "=":
                r["lhs"] = r["rhs"] = constrs[rows.index(r)].RHS

        for round_idx in range(self.max_rounds):
            tightened_this_round = 0

            for row in rows:
                coeffs = row["coeffs"]
                lhs    = row["lhs"]
                rhs    = row["rhs"]

                for j, (idx, a) in enumerate(coeffs):
                    if abs(a) <= self.eps:
                        continue

                    rest_min = 0.0
                    rest_max = 0.0
                    bounded  = True

                    for k, (other_idx, b) in enumerate(coeffs):
                        if k == j:
                            continue
                        if b > 0:
                            if math.isinf(lb[other_idx]):
                                rest_min = -math.inf
                                bounded  = False
                            else:
                                rest_min += b * lb[other_idx]
                            if math.isinf(ub[other_idx]):
                                rest_max = math.inf
                                bounded  = False
                            else:
                                rest_max += b * ub[other_idx]
                        else:
                            if math.isinf(ub[other_idx]):
                                rest_min = -math.inf
                                bounded  = False
                            else:
                                rest_min += b * ub[other_idx]
                            if math.isinf(lb[other_idx]):
                                rest_max = math.inf
                                bounded  = False
                            else:
                                rest_max += b * lb[other_idx]

                    new_lb = lb[idx]
                    new_ub = ub[idx]

                    if math.isfinite(lhs) and math.isfinite(rest_max):
                        implied = (lhs - rest_max) / a
                        if a > 0:
                            new_lb = max(new_lb, implied)
                        else:
                            new_ub = min(new_ub, implied)

                    if math.isfinite(rhs) and math.isfinite(rest_min):
                        implied = (rhs - rest_min) / a
                        if a > 0:
                            new_ub = min(new_ub, implied)
                        else:
                            new_lb = max(new_lb, implied)

                    if vtype[idx] in (GRB.BINARY, GRB.INTEGER):
                        if math.isfinite(new_lb):
                            new_lb = math.ceil(new_lb - self.eps)
                        if math.isfinite(new_ub):
                            new_ub = math.floor(new_ub + self.eps)

                    if new_lb > new_ub + self.eps:
                        result.infeasible = True
                        if self.verbose:
                            print(f"  [BoundTightening] INVIÁVEL: var {idx} "
                                  f"lb={new_lb:.6f} > ub={new_ub:.6f}")
                        return result

                    if new_lb > lb[idx] + self.eps:
                        lb[idx]              = new_lb
                        tightened_this_round += 1
                        result.bounds_tightened  += 1

                    if new_ub < ub[idx] - self.eps:
                        ub[idx]              = new_ub
                        tightened_this_round += 1
                        result.bounds_tightened  += 1

            if self.verbose:
                print(f"  [BoundTightening] rodada {round_idx + 1}: "
                      f"{tightened_this_round} bounds apertados")

            if tightened_this_round == 0:
                break

        for i, v in enumerate(vars_list):
            if lb[i] > v.LB + self.eps:
                v.LB = lb[i]
            if ub[i] < v.UB - self.eps:
                v.UB = ub[i]

            if abs(ub[i] - lb[i]) <= self.eps:
                result.vars_fixed += 1

        model.update()
        return result


class RowFeasibilityAndRedundancy(PresolveRule):
    def __init__(self, eps: float = 1e-8, verbose: bool = False):
        self.eps     = eps
        self.verbose = verbose

    def apply(self, model: gp.Model) -> PresolveResult:
        result    = PresolveResult()
        vars_list = model.getVars()
        constrs   = model.getConstrs()

        var_id_to_idx = {id(v): i for i, v in enumerate(vars_list)}

        lb = [v.LB for v in vars_list]
        ub = [v.UB for v in vars_list]

        redundant = []  

        for c in constrs:
            row_gurobi = model.getRow(c)
            sense      = c.Sense
            rhs        = c.RHS

            if sense == "<":
                row_lhs = -math.inf
                row_rhs = rhs
            elif sense == ">":
                row_lhs = rhs
                row_rhs = math.inf
            else:  
                row_lhs = rhs
                row_rhs = rhs

            min_act =  0.0
            max_act =  0.0
            bounded =  True

            for k in range(row_gurobi.size()):
                v    = row_gurobi.getVar(k)
                idx  = var_id_to_idx.get(id(v), -1)
                if idx < 0:
                    continue
                a = row_gurobi.getCoeff(k)

                if a > 0:
                    if math.isinf(lb[idx]):
                        min_act = -math.inf
                        bounded = False
                    else:
                        min_act += a * lb[idx]
                    if math.isinf(ub[idx]):
                        max_act = math.inf
                        bounded = False
                    else:
                        max_act += a * ub[idx]
                else:
                    if math.isinf(ub[idx]):
                        min_act = -math.inf
                        bounded = False
                    else:
                        min_act += a * ub[idx]
                    if math.isinf(lb[idx]):
                        max_act = math.inf
                        bounded = False
                    else:
                        max_act += a * lb[idx]

            if math.isfinite(row_lhs) and math.isfinite(max_act):
                if max_act < row_lhs - self.eps:
                    result.infeasible = True
                    if self.verbose:
                        print(f"  [RowFeasibility] INVIÁVEL: '{c.ConstrName}' "
                              f"max_act={max_act:.6f} < lhs={row_lhs:.6f}")
                    return result

            if math.isfinite(row_rhs) and math.isfinite(min_act):
                if min_act > row_rhs + self.eps:
                    result.infeasible = True
                    if self.verbose:
                        print(f"  [RowFeasibility] INVIÁVEL: '{c.ConstrName}' "
                              f"min_act={min_act:.6f} > rhs={row_rhs:.6f}")
                    return result

            lhs_ok = (not math.isfinite(row_lhs)) or (
                math.isfinite(min_act) and min_act >= row_lhs - self.eps
            )
            rhs_ok = (not math.isfinite(row_rhs)) or (
                math.isfinite(max_act) and max_act <= row_rhs + self.eps
            )

            if lhs_ok and rhs_ok:
                redundant.append(c)
                if self.verbose:
                    print(f"  [RowFeasibility] redundante: '{c.ConstrName}' "
                          f"[{min_act:.4f}, {max_act:.4f}] ⊆ "
                          f"[{row_lhs:.4f}, {row_rhs:.4f}]")

        if redundant:
            model.remove(redundant)
            model.update()
            result.rows_removed = len(redundant)
            if self.verbose:
                print(f"  [RowFeasibility] {len(redundant)} restrição(ões) removida(s)")

        return result


class SubstituteFixedVariables(PresolveRule):
    def __init__(self, eps: float = 1e-8, verbose: bool = False):
        self.eps     = eps
        self.verbose = verbose

    def apply(self, model: gp.Model) -> PresolveResult:
        result    = PresolveResult()
        vars_list = model.getVars()
        constrs   = model.getConstrs()

        fixed = {}  
        for i, v in enumerate(vars_list):
            if abs(v.UB - v.LB) <= self.eps:
                fixed[i] = v.LB

        if not fixed:
            if self.verbose:
                print("  [SubstituteFixed] nenhuma variável fixa encontrada")
            return result

        if self.verbose:
            print(f"  [SubstituteFixed] {len(fixed)} variável(is) fixa(s) encontrada(s)")

        var_id_to_idx = {id(v): i for i, v in enumerate(vars_list)}

        substitutions = 0  
        empty_rows    = 0  

        for c in constrs:
            row_gurobi = model.getRow(c)
            sense      = c.Sense

            shift      = 0.0
            has_fixed  = False

            for k in range(row_gurobi.size()):
                v   = row_gurobi.getVar(k)
                idx = var_id_to_idx.get(id(v), -1)
                if idx in fixed:
                    shift     += row_gurobi.getCoeff(k) * fixed[idx]
                    has_fixed  = True

            if not has_fixed:
                continue

            new_rhs = c.RHS - shift
            c.RHS   = new_rhs
            substitutions += 1

            if self.verbose:
                print(f"  [SubstituteFixed] '{c.ConstrName}': "
                      f"shift={shift:.6f} → novo RHS={new_rhs:.6f}")

            n_free = sum(
                1 for k in range(row_gurobi.size())
                if var_id_to_idx.get(id(row_gurobi.getVar(k)), -1) not in fixed
            )
            if n_free == 0:
                empty_rows += 1
                if self.verbose:
                    print(f"  [SubstituteFixed] '{c.ConstrName}' ficou vazia "
                          f"após substituição")

        model.update()

        result.vars_fixed = len(fixed)
        result.coefficients_changed = substitutions
        if self.verbose:
            print(f"  [SubstituteFixed] {substitutions} substituição(ões) em "
                  f"{len(constrs)} restrição(ões), {empty_rows} linha(s) vazia(s)")

        return result


class ImplicitFreeVariableDetection(PresolveRule):
    def __init__(self, eps: float = 1e-8, relax: bool = False, verbose: bool = False):
        self.eps     = eps
        self.relax   = relax
        self.verbose = verbose

    def apply(self, model: gp.Model) -> PresolveResult:
        result    = PresolveResult()
        vars_list = model.getVars()
        constrs   = model.getConstrs()

        var_id_to_idx = {id(v): i for i, v in enumerate(vars_list)}

        lb = [v.LB for v in vars_list]
        ub = [v.UB for v in vars_list]

        rows_data = []
        for c in constrs:
            row_gurobi = model.getRow(c)
            coeffs = []
            for k in range(row_gurobi.size()):
                v   = row_gurobi.getVar(k)
                idx = var_id_to_idx.get(id(v), -1)
                if idx >= 0:
                    coeffs.append((idx, row_gurobi.getCoeff(k)))

            if c.Sense == "<":
                row_lhs = -math.inf
                row_rhs = c.RHS
            elif c.Sense == ">":
                row_lhs = c.RHS
                row_rhs = math.inf
            else:
                row_lhs = c.RHS
                row_rhs = c.RHS

            rows_data.append({
                "coeffs": coeffs,
                "lhs":    row_lhs,
                "rhs":    row_rhs,
            })

        var_to_rows = {i: [] for i in range(len(vars_list))}
        for row_idx, row in enumerate(rows_data):
            for var_idx, _ in row["coeffs"]:
                var_to_rows[var_idx].append(row_idx)

        implied_free_count = 0
        relaxed_count      = 0

        for var_idx, v in enumerate(vars_list):
            if abs(ub[var_idx] - lb[var_idx]) <= self.eps:
                continue

            if math.isinf(lb[var_idx]) and math.isinf(ub[var_idx]):
                continue

            row_lb = -math.inf
            row_ub =  math.inf

            for row_idx in var_to_rows[var_idx]:
                row   = rows_data[row_idx]
                a     = next(coef for idx, coef in row["coeffs"] if idx == var_idx)
                lhs   = row["lhs"]
                rhs   = row["rhs"]

                rest_min = 0.0
                rest_max = 0.0

                for other_idx, b in row["coeffs"]:
                    if other_idx == var_idx:
                        continue
                    if b > 0:
                        rest_min += b * lb[other_idx] if not math.isinf(lb[other_idx]) else -math.inf
                        rest_max += b * ub[other_idx] if not math.isinf(ub[other_idx]) else  math.inf
                    else:
                        rest_min += b * ub[other_idx] if not math.isinf(ub[other_idx]) else -math.inf
                        rest_max += b * lb[other_idx] if not math.isinf(lb[other_idx]) else  math.inf

                    if math.isinf(rest_min) or math.isinf(rest_max):
                        break

                if math.isinf(rest_min) or math.isinf(rest_max):
                    continue  

                if a > 0:
                    if math.isfinite(lhs):
                        row_lb = max(row_lb, (lhs - rest_max) / a)
                    if math.isfinite(rhs):
                        row_ub = min(row_ub, (rhs - rest_min) / a)
                else: 
                    if math.isfinite(lhs):
                        row_ub = min(row_ub, (lhs - rest_max) / a)
                    if math.isfinite(rhs):
                        row_lb = max(row_lb, (rhs - rest_min) / a)

            lb_implied = math.isfinite(row_lb) and row_lb >= lb[var_idx] - self.eps
            ub_implied = math.isfinite(row_ub) and row_ub <= ub[var_idx] + self.eps

            is_implied_free = lb_implied and ub_implied

            if is_implied_free:
                implied_free_count += 1
                if self.verbose:
                    print(f"  [ImplicitFree] var {var_idx} ('{v.VarName}'): "
                          f"explicit=[{lb[var_idx]:.4f}, {ub[var_idx]:.4f}] "
                          f"implied=[{row_lb:.4f}, {row_ub:.4f}] → implied free")

                if self.relax:
                    if lb_implied and not math.isinf(lb[var_idx]):
                        vars_list[var_idx].LB = (
                            -math.inf if vars_list[var_idx].VType == GRB.CONTINUOUS else lb[var_idx]
                        )
                        relaxed_count += 1
                        result.bounds_tightened += 1

                    if ub_implied and not math.isinf(ub[var_idx]):
                        vars_list[var_idx].UB = (
                            math.inf if vars_list[var_idx].VType == GRB.CONTINUOUS else ub[var_idx]
                        )
                        relaxed_count += 1
                        result.bounds_tightened += 1

        if self.verbose:
            print(f"  [ImplicitFree] {implied_free_count} variável(is) implied free detectada(s)"
                  + (f", {relaxed_count} bound(s) relaxado(s)" if self.relax else ""))

        if relaxed_count > 0:
            model.update()

        return result


class SingletonRowReduction(PresolveRule):

    def __init__(self, eps: float = 1e-8, verbose: bool = False):
        self.eps     = eps
        self.verbose = verbose

    def apply(self, model: gp.Model) -> PresolveResult:
        result    = PresolveResult()
        vars_list = model.getVars()
        constrs   = model.getConstrs()

        var_id_to_idx = {id(v): i for i, v in enumerate(vars_list)}
        vtype         = [v.VType for v in vars_list]

        singletons = [] 

        for c in constrs:
            row_gurobi = model.getRow(c)

            if row_gurobi.size() != 1:
                continue

            v   = row_gurobi.getVar(0)
            idx = var_id_to_idx.get(id(v), -1)
            if idx < 0:
                continue

            a     = row_gurobi.getCoeff(0)
            sense = c.Sense
            rhs   = c.RHS

            if sense == "<":
                row_lhs = -math.inf
                row_rhs = rhs
            elif sense == ">":
                row_lhs = rhs
                row_rhs = math.inf
            else:  
                row_lhs = rhs
                row_rhs = rhs

            if a > 0:
                implied_lb = row_lhs / a if math.isfinite(row_lhs) else -math.inf
                implied_ub = row_rhs / a if math.isfinite(row_rhs) else  math.inf
            else: 
                implied_lb = row_rhs / a if math.isfinite(row_rhs) else -math.inf
                implied_ub = row_lhs / a if math.isfinite(row_lhs) else  math.inf

            is_int = vtype[idx] in (GRB.BINARY, GRB.INTEGER)
            if is_int:
                if math.isfinite(implied_lb):
                    implied_lb = math.ceil(implied_lb - self.eps)
                if math.isfinite(implied_ub):
                    implied_ub = math.floor(implied_ub + self.eps)

            tightened = False

            if math.isfinite(implied_lb) and implied_lb > vars_list[idx].LB + self.eps:
                if self.verbose:
                    print(f"  [SingletonRow] '{c.ConstrName}': "
                          f"var '{v.VarName}' lb {vars_list[idx].LB:.4f} → {implied_lb:.4f}")
                vars_list[idx].LB = implied_lb
                tightened = True
                result.bounds_tightened += 1

            if math.isfinite(implied_ub) and implied_ub < vars_list[idx].UB - self.eps:
                if self.verbose:
                    print(f"  [SingletonRow] '{c.ConstrName}': "
                          f"var '{v.VarName}' ub {vars_list[idx].UB:.4f} → {implied_ub:.4f}")
                vars_list[idx].UB = implied_ub
                tightened = True
                result.bounds_tightened += 1

            if vars_list[idx].LB > vars_list[idx].UB + self.eps:
                result.infeasible = True
                if self.verbose:
                    print(f"  [SingletonRow] INVIÁVEL: var '{v.VarName}' "
                          f"lb={vars_list[idx].LB:.6f} > ub={vars_list[idx].UB:.6f}")
                return result

            singletons.append(c)

        if singletons:
            model.remove(singletons)
            model.update()
            result.rows_removed = len(singletons)
            if self.verbose:
                print(f"  [SingletonRow] {len(singletons)} singleton row(s) removida(s)")

        return result


class EuclideanReduction(PresolveRule):

    def __init__(self, eps: float = 1e-6, verbose: bool = False):
        self.eps     = eps
        self.verbose = verbose

    @staticmethod
    def _gcd_list(values: list) -> int:
        """Calcula o MDC de uma lista de inteiros positivos."""
        from math import gcd
        from functools import reduce
        return reduce(gcd, values)

    def apply(self, model: gp.Model) -> PresolveResult:
        from math import gcd
        result    = PresolveResult()
        vars_list = model.getVars()
        constrs   = model.getConstrs()

        var_id_to_idx = {id(v): i for i, v in enumerate(vars_list)}
        vtype         = [v.VType for v in vars_list]

        reductions = 0

        for c in constrs:
            row_gurobi = model.getRow(c)
            sense      = c.Sense
            rhs        = c.RHS

            if row_gurobi.size() == 0:
                continue

            all_int = True
            coeffs  = []
            for k in range(row_gurobi.size()):
                v   = row_gurobi.getVar(k)
                idx = var_id_to_idx.get(id(v), -1)
                if idx < 0:
                    all_int = False
                    break
                if vtype[idx] not in (GRB.BINARY, GRB.INTEGER):
                    all_int = False
                    break
                a = row_gurobi.getCoeff(k)
                if abs(a - round(a)) > self.eps:
                    all_int = False
                    break
                coeffs.append((idx, int(round(a))))

            if not all_int or not coeffs:
                continue

            abs_coeffs = [abs(a) for _, a in coeffs]
            g = self._gcd_list(abs_coeffs)

            if g <= 1:
                continue  

            if abs(rhs - round(rhs)) > self.eps:
                continue  
            rhs_int = int(round(rhs))

            is_equality = (sense == "=")

            if is_equality:
                if rhs_int % g != 0:
                    result.infeasible = True
                    if self.verbose:
                        print(f"  [EuclideanReduction] INVIÁVEL: '{c.ConstrName}' "
                              f"rhs={rhs_int} não divisível por g={g}")
                    return result
                new_rhs = rhs_int // g
                c.RHS   = float(new_rhs)
                for k in range(row_gurobi.size()):
                    pass  

            else:
                if sense == "<":
                    new_rhs = math.floor(rhs / g)
                else:  
                    new_rhs = math.ceil(rhs / g)

                if new_rhs != rhs_int:
                    c.RHS = float(new_rhs * g) 

                    if self.verbose:
                        print(f"  [EuclideanReduction] '{c.ConstrName}': "
                              f"g={g}, rhs {rhs_int} → {new_rhs * g} "
                              f"(equivalente a rhs/g={new_rhs})")
                    result.coefficients_changed  += 1
                    reductions += 1
                    continue

            new_coeffs = [(idx, a // g) for idx, a in coeffs]
            new_rhs_f  = float(rhs_int // g) if is_equality else float(
                math.floor(rhs / g) if sense == "<" else math.ceil(rhs / g)
            )

            new_expr = gp.LinExpr()
            for idx, new_a in new_coeffs:
                new_expr += float(new_a) * vars_list[idx]

            constr_name = c.ConstrName
            model.remove(c)
            model.addConstr(
                new_expr <= new_rhs_f if sense == "<" else (
                new_expr >= new_rhs_f if sense == ">" else
                new_expr == new_rhs_f
                ),
                name=constr_name
            )

            result.coefficients_changed  += 1
            reductions += 1

            if self.verbose:
                print(f"  [EuclideanReduction] '{constr_name}': "
                      f"g={g}, dividido por {g} → rhs={new_rhs_f:.0f}")

        if reductions > 0:
            model.update()

        if self.verbose:
            print(f"  [EuclideanReduction] {reductions} restrição(ões) reduzida(s)")

        return result


class CoefficientStrengthening(PresolveRule):

    def __init__(self, eps: float = 1e-8, verbose: bool = False):
        self.eps     = eps
        self.verbose = verbose

    def apply(self, model: gp.Model) -> PresolveResult:
        result    = PresolveResult()
        vars_list = model.getVars()
        constrs   = list(model.getConstrs())

        var_id_to_idx = {id(v): i for i, v in enumerate(vars_list)}
        vtype         = [v.VType for v in vars_list]
        lb            = [v.LB    for v in vars_list]
        ub            = [v.UB    for v in vars_list]

        strengthened = 0

        for c in constrs:
            if c.Sense != "<":
                continue

            rhs        = c.RHS
            row_gurobi = model.getRow(c)

            if row_gurobi.size() == 0:
                continue

            coeffs = []
            for k in range(row_gurobi.size()):
                v   = row_gurobi.getVar(k)
                idx = var_id_to_idx.get(id(v), -1)
                if idx < 0:
                    continue
                coeffs.append((idx, row_gurobi.getCoeff(k)))

            modified    = False
            new_coeffs  = dict(coeffs) 
            new_rhs     = rhs

            for j, (idx, a) in enumerate(coeffs):
                if vtype[idx] not in (GRB.BINARY, GRB.INTEGER):
                    continue

                rest_max = 0.0
                bounded  = True
                for k2, (other_idx, b) in enumerate(coeffs):
                    if other_idx == idx:
                        continue
                    if b > 0:
                        if math.isinf(ub[other_idx]):
                            bounded = False
                            break
                        rest_max += b * ub[other_idx]
                    else:
                        if math.isinf(lb[other_idx]):
                            bounded = False
                            break
                        rest_max += b * lb[other_idx]

                if not bounded:
                    continue

                if a > 0 and math.isfinite(ub[idx]):
                    d = new_rhs - rest_max - a * (ub[idx] - 1)
                    if d > self.eps and a >= d - self.eps:
                        new_a       = a - d
                        new_rhs    -= d * ub[idx]
                        new_coeffs[idx] = new_a
                        modified    = True
                        if self.verbose:
                            print(f"  [CoeffStrength] '{c.ConstrName}': "
                                  f"var {idx} coef {a:.4f} → {new_a:.4f}, "
                                  f"rhs {rhs:.4f} → {new_rhs:.4f} (d={d:.4f})")

                elif a < 0 and math.isfinite(lb[idx]):
                    d = new_rhs - rest_max - a * (lb[idx] + 1)
                    if d > self.eps and -a >= d - self.eps:
                        new_a       = a + d
                        new_rhs    += d * lb[idx]
                        new_coeffs[idx] = new_a
                        modified    = True
                        if self.verbose:
                            print(f"  [CoeffStrength] '{c.ConstrName}': "
                                  f"var {idx} coef {a:.4f} → {new_a:.4f}, "
                                  f"rhs {rhs:.4f} → {new_rhs:.4f} (d={d:.4f})")

            if not modified:
                continue

            constr_name = c.ConstrName
            new_expr    = gp.LinExpr()
            for idx, new_a in new_coeffs.items():
                new_expr += float(new_a) * vars_list[idx]

            model.remove(c)
            model.addConstr(new_expr <= new_rhs, name=constr_name)
            result.coefficients_changed += 1
            strengthened += 1

        if strengthened > 0:
            model.update()

        if self.verbose:
            print(f"  [CoeffStrength] {strengthened} restrição(ões) fortalecida(s)")

        return result


class ChvatalGomoryStrengthening(PresolveRule):
    def __init__(self, eps: float = 1e-8, max_scales: int = 5, verbose: bool = False):
        self.eps        = eps
        self.max_scales = max_scales
        self.verbose    = verbose

    def apply(self, model: gp.Model) -> PresolveResult:
        result    = PresolveResult()
        vars_list = model.getVars()
        constrs   = list(model.getConstrs())

        var_id_to_idx = {id(v): i for i, v in enumerate(vars_list)}
        vtype         = [v.VType for v in vars_list]
        lb_arr        = [v.LB    for v in vars_list]

        strengthened = 0

        for c in constrs:
            if c.Sense != ">":
                continue

            bi         = c.RHS
            row_gurobi = model.getRow(c)

            if bi <= self.eps:
                continue

            if row_gurobi.size() == 0:
                continue

            coeffs  = []
            all_ok  = True
            for k in range(row_gurobi.size()):
                v   = row_gurobi.getVar(k)
                idx = var_id_to_idx.get(id(v), -1)
                if idx < 0:
                    all_ok = False
                    break
                if vtype[idx] not in (GRB.BINARY, GRB.INTEGER):
                    all_ok = False
                    break
                if lb_arr[idx] < -self.eps:
                    all_ok = False
                    break
                coeffs.append((idx, row_gurobi.getCoeff(k)))

            if not all_ok or not coeffs:
                continue

            abs_vals = [abs(a) for _, a in coeffs if abs(a) > self.eps]
            if not abs_vals:
                continue

            amax = max(abs_vals)
            amin = min(abs_vals)

            scalars = {1.0}
            for t in range(1, self.max_scales + 1):
                if amax > self.eps:
                    scalars.add(t / amax)
                if amin > self.eps:
                    scalars.add(t / amin)
                    scalars.add((2 * t - 1) / (2 * amin))

            best        = None
            best_ratio  = math.inf

            for s in scalars:
                if s <= self.eps:
                    continue

                new_coeffs = {idx: math.ceil(a * s - 1e-6) for idx, a in coeffs}
                new_b      = math.ceil(bi * s - 1e-6)

                if new_b <= self.eps:
                    continue

                tighter = any(
                    new_coeffs[idx] * bi < a * new_b - self.eps
                    for idx, a in coeffs
                )
                if not tighter:
                    continue

                total_coeffs = sum(new_coeffs.values())
                ratio        = total_coeffs / new_b if new_b > 0 else math.inf

                if ratio < best_ratio:
                    best_ratio = ratio
                    best       = (new_coeffs, new_b)

            if best is None:
                continue

            new_coeffs_dict, new_b = best

            constr_name = c.ConstrName
            new_expr    = gp.LinExpr()
            for idx, new_a in new_coeffs_dict.items():
                new_expr += float(new_a) * vars_list[idx]

            model.remove(c)
            model.addConstr(new_expr >= float(new_b), name=constr_name)

            result.coefficients_changed += 1
            strengthened += 1

            if self.verbose:
                orig_str = " + ".join(f"{a:.2f}*x{idx}" for idx, a in coeffs)
                new_str  = " + ".join(f"{a:.2f}*x{idx}" for idx, a in new_coeffs_dict.items())
                print(f"  [CG-Strength] '{constr_name}': "
                      f"{orig_str} >= {bi:.2f}  →  "
                      f"{new_str} >= {new_b:.2f}")

        if strengthened > 0:
            model.update()

        if self.verbose:
            print(f"  [CG-Strength] {strengthened} restrição(ões) fortalecida(s)")

        return result


class ParallelRows(PresolveRule):

    def __init__(self, eps: float = 1e-8, verbose: bool = False):
        self.eps     = eps
        self.verbose = verbose

    def _proportionality_factor(self, coeffs1: list, coeffs2: list) -> Optional[float]:
        if len(coeffs1) != len(coeffs2):
            return None

        d1 = {idx: a for idx, a in coeffs1}
        d2 = {idx: a for idx, a in coeffs2}

        if set(d1.keys()) != set(d2.keys()):
            return None

        lam = None
        for idx in d1:
            a1 = d1[idx]
            a2 = d2[idx]
            if abs(a1) <= self.eps:
                if abs(a2) > self.eps:
                    return None  
                continue  
            if abs(a2) <= self.eps:
                return None 
            ratio = a2 / a1
            if lam is None:
                lam = ratio
            elif abs(ratio - lam) > self.eps * max(1.0, abs(lam)):
                return None 

        return lam  

    def apply(self, model: gp.Model) -> PresolveResult:
        result    = PresolveResult()
        vars_list = model.getVars()
        constrs   = model.getConstrs()

        var_id_to_idx = {id(v): i for i, v in enumerate(vars_list)}

        rows_data = []
        for c in constrs:
            row_gurobi = model.getRow(c)
            coeffs = []
            for k in range(row_gurobi.size()):
                v   = row_gurobi.getVar(k)
                idx = var_id_to_idx.get(id(v), -1)
                if idx >= 0:
                    coeffs.append((idx, row_gurobi.getCoeff(k)))
            coeffs.sort(key=lambda x: x[0]) 

            sense = c.Sense
            rhs   = c.RHS
            if sense == "<":
                lhs_val = -math.inf
                rhs_val = rhs
            elif sense == ">":
                lhs_val = rhs
                rhs_val = math.inf
            else:
                lhs_val = rhs
                rhs_val = rhs

            rows_data.append({
                "constr": c,
                "coeffs": coeffs,
                "lhs":    lhs_val,
                "rhs":    rhs_val,
                "remove": False,
            })

        removed   = 0
        n         = len(rows_data)

        for i in range(n):
            if rows_data[i]["remove"]:
                continue

            r1 = rows_data[i]

            for j in range(i + 1, n):
                if rows_data[j]["remove"]:
                    continue

                r2 = rows_data[j]

                lam = self._proportionality_factor(r1["coeffs"], r2["coeffs"])
                if lam is None or abs(lam) <= self.eps:
                    continue

                r2_lhs = r2["lhs"]
                r2_rhs = r2["rhs"]

                if lam > 0:
                    r2_lhs_norm = r2_lhs / lam if math.isfinite(r2_lhs) else -math.inf
                    r2_rhs_norm = r2_rhs / lam if math.isfinite(r2_rhs) else  math.inf

                    new_lhs = max(r1["lhs"], r2_lhs_norm)
                    new_rhs = min(r1["rhs"], r2_rhs_norm)

                    if new_lhs > new_rhs + self.eps:
                        result.infeasible = True
                        if self.verbose:
                            print(f"  [ParallelRows] INVIÁVEL: '{r1['constr'].ConstrName}' "
                                  f"e '{r2['constr'].ConstrName}' — intersecção vazia "
                                  f"[{new_lhs:.4f}, {new_rhs:.4f}]")
                        return result

                    if self.verbose:
                        print(f"  [ParallelRows] '{r1['constr'].ConstrName}' absorve "
                              f"'{r2['constr'].ConstrName}' (λ={lam:.4f}): "
                              f"[{r1['lhs']:.4f},{r1['rhs']:.4f}] ∩ "
                              f"[{r2_lhs_norm:.4f},{r2_rhs_norm:.4f}] = "
                              f"[{new_lhs:.4f},{new_rhs:.4f}]")

                    r1["lhs"] = new_lhs
                    r1["rhs"] = new_rhs
                    rows_data[j]["remove"] = True
                    removed += 1

                else:  
                    r2_lhs_norm = r2_rhs / lam if math.isfinite(r2_rhs) else -math.inf
                    r2_rhs_norm = r2_lhs / lam if math.isfinite(r2_lhs) else  math.inf

                    combined_lhs = max(r1["lhs"], r2_lhs_norm)
                    combined_rhs = min(r1["rhs"], r2_rhs_norm)

                    if combined_lhs > combined_rhs + self.eps:
                        result.infeasible = True
                        if self.verbose:
                            print(f"  [ParallelRows] INVIÁVEL: '{r1['constr'].ConstrName}' "
                                  f"e '{r2['constr'].ConstrName}' — combinação vazia "
                                  f"(λ={lam:.4f})")
                        return result

                    if self.verbose:
                        print(f"  [ParallelRows] '{r1['constr'].ConstrName}' + "
                              f"'{r2['constr'].ConstrName}' (λ={lam:.4f}) → "
                              f"ranged [{combined_lhs:.4f}, {combined_rhs:.4f}]")

                    r1["lhs"] = combined_lhs
                    r1["rhs"] = combined_rhs
                    rows_data[j]["remove"] = True
                    removed += 1

        if removed == 0:
            return result

        to_remove = []
        for row in rows_data:
            c       = row["constr"]
            new_lhs = row["lhs"]
            new_rhs = row["rhs"]

            if row["remove"]:
                to_remove.append(c)
                continue

            orig_rhs   = c.RHS
            orig_sense = c.Sense

            if math.isfinite(new_lhs) and math.isfinite(new_rhs):
                if abs(new_lhs - new_rhs) <= self.eps:
                    if orig_sense != "=" or abs(orig_rhs - new_rhs) > self.eps:
                        c.RHS = new_rhs
                        result.coefficients_changed  += 1
                elif not math.isfinite(new_lhs):
                    if orig_sense != "<" or abs(orig_rhs - new_rhs) > self.eps:
                        c.RHS = new_rhs
                        result.coefficients_changed  += 1
                elif not math.isfinite(new_rhs):
                    if orig_sense != ">" or abs(orig_rhs - new_lhs) > self.eps:
                        c.RHS = new_lhs
                        result.coefficients_changed  += 1
                else:
                    if orig_sense == "<" and abs(orig_rhs - new_rhs) > self.eps:
                        c.RHS = new_rhs
                        result.coefficients_changed  += 1
                    elif orig_sense == ">" and abs(orig_rhs - new_lhs) > self.eps:
                        c.RHS = new_lhs
                        result.coefficients_changed  += 1
            elif math.isfinite(new_rhs):
                if orig_sense != "<" or abs(orig_rhs - new_rhs) > self.eps:
                    c.RHS = new_rhs
                    result.coefficients_changed  += 1
            elif math.isfinite(new_lhs):
                if orig_sense != ">" or abs(orig_rhs - new_lhs) > self.eps:
                    c.RHS = new_lhs
                    result.coefficients_changed  += 1

        if to_remove:
            model.remove(to_remove)
            result.rows_removed = len(to_remove)

        model.update()

        if self.verbose:
            print(f"  [ParallelRows] {removed} restrição(ões) dominada(s) removida(s)")

        return result


class Presolver:

    def __init__(self, rules: list = None, verbose: bool = True):
        self.rules   = rules or [BoundTightening()]
        self.verbose = verbose

    def run(self, model: gp.Model) -> tuple[bool, list]:
        if self.verbose:
            print(f"  [Presolver] aplicando {len(self.rules)} regra(s)...")

        total_tightened = 0
        history = []

        for rule in self.rules:
            before_vars = model.NumVars
            before_cons = model.NumConstrs

            t0 = time.perf_counter()
            result = rule.apply(model)
            result.runtime = time.perf_counter() - t0

            after_vars = model.NumVars
            after_cons = model.NumConstrs

            total_tightened += result.bounds_tightened

            history.append({
                "rule": repr(rule),
                "before_vars": before_vars,
                "before_constraints": before_cons,
                "after_vars": after_vars,
                "after_constraints": after_cons,
                **result.to_dict()
            })

            if result.infeasible:
                if self.verbose:
                    print(f"  [Presolver] {rule} detectou INVIABILIDADE.")
                return False, history

            if self.verbose:
                print(
                    f"  [Presolver] {rule}: "
                    f"bounds={result.bounds_tightened}, "
                    f"rows_removed={result.rows_removed}, "
                    f"vars_fixed={result.vars_fixed}, "
                    f"coeff_changes={result.coefficients_changed}, "
                    f"time={result.runtime:.4f}s"
                )

        if self.verbose:
            total_rows_removed = sum(h["rows_removed"] for h in history)
            total_vars_fixed = sum(h["vars_fixed"] for h in history)
            total_coeff_changes = sum(h["coefficients_changed"] for h in history)

            print(
                f"  [Presolver] total: "
                f"bounds={total_tightened}, "
                f"rows_removed={total_rows_removed}, "
                f"vars_fixed={total_vars_fixed}, "
                f"coeff_changes={total_coeff_changes}"
            )

        return True, history

    def __repr__(self) -> str:
        return f"Presolver(rules={self.rules})"
