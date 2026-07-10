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


@dataclass
class _CutRow:
    basic_col: int
    b_value: float
    f0: float
    entries: list
    bound_width: dict
    other_bound: dict
    A_dense: np.ndarray
    b_vec: np.ndarray
    n: int
 
 
class CuttingPlane(ABC):
    def __init__(self, only_root: bool = True, max_cuts: int = 20,
                 tol: float = 1e-6, probability: float = 1.0):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability deve estar em [0, 1], recebido: {probability}")
        self.only_root   = only_root
        self.max_cuts    = max_cuts
        self.tol         = tol
        self.probability = probability
 
    def try_run(self, node, bb) -> int:
        """Verifica elegibilidade e sorteia antes de delegar para run()."""
        if not self.should_run(node, bb):
            return 0
        if random.random() >= self.probability:
            return 0
        return self.run(node, bb)
 
    def should_run(self, node, bb) -> bool:
        if self.only_root:
            return node.depth == 0
        return True
 
    def run(self, node, bb) -> int:
        rows  = self._extract_fractional_rows(node, bb)
        added = 0
        for row in rows[: self.max_cuts]:
            cut = self.generate_cut(row, bb)
            if cut is None:
                continue
            coeffs, rhs, sense = cut
            if self._add_cut_to_model(coeffs, rhs, sense, bb):
                added += 1
        return added
 
    @abstractmethod
    def generate_cut(self, row: _CutRow, bb):
        pass
 
    def _extract_fractional_rows(self, node, bb) -> list:
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
        if model.Status != GRB.OPTIMAL:
            return []

        n       = bb._n
        constrs = model.getConstrs()
        m       = len(constrs)
        if m == 0:
            return []

        A = model.getA().toarray()

        slack_lb   = np.zeros(m)
        slack_ub   = np.zeros(m)
        b_vec      = np.zeros(m)
        slack_sense = []
        for i, c in enumerate(constrs):
            b_vec[i] = c.RHS
            slack_sense.append(c.Sense)
            if c.Sense == '<':
                slack_lb[i], slack_ub[i] = 0.0, math.inf
            elif c.Sense == '>':
                slack_lb[i], slack_ub[i] = -math.inf, 0.0
            else:   
                slack_lb[i], slack_ub[i] = 0.0, 0.0

        A_full = np.hstack([A, np.eye(m)])
        total_cols = n + m

        lb_full = np.array([v.LB for v in vars_list] + list(slack_lb))
        ub_full = np.array([v.UB for v in vars_list] + list(slack_ub))

        status = [v.VBasis for v in vars_list] + [c.CBasis for c in constrs]

        basic_cols = [j for j in range(total_cols) if status[j] == 0]
        if len(basic_cols) != m:
            return []  

        try:
            B    = A_full[:, basic_cols]
            Binv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return []

        nonbasic_cols = [j for j in range(total_cols) if status[j] != 0]
        x_rest   = np.zeros(total_cols)
        at_upper = {}

        for j in nonbasic_cols:
            if j >= n:
                si = j - n
                sense = slack_sense[si]
                if sense == '<':
                    x_rest[j]   = 0.0  
                    at_upper[j] = False
                elif sense == '>':
                    x_rest[j]   = 0.0  
                    at_upper[j] = True
                else:  
                    x_rest[j]   = 0.0
                    at_upper[j] = False
                continue

            if status[j] == -2:
                x_rest[j]   = ub_full[j]
                at_upper[j] = True
            elif status[j] == -3:
                x_rest[j]   = 0.0
                at_upper[j] = False
            else:   
                x_rest[j]   = lb_full[j]
                at_upper[j] = False

        A_N = A_full[:, nonbasic_cols]

        sign       = np.array([-1.0 if at_upper.get(j, False) else 1.0 for j in nonbasic_cols])
        A_N_signed = A_N * sign[np.newaxis, :]

        x_B  = Binv @ (b_vec - A_full[:, nonbasic_cols] @ x_rest[nonbasic_cols])
        Tsig = Binv @ A_N_signed

        basic_row_of = {col: idx for idx, col in enumerate(basic_cols)}

        rows = []
        for col0 in basic_cols:
            if col0 >= n or col0 not in bb._integer_vars:
                continue
            row_idx = basic_row_of[col0]
            b_val   = x_B[row_idx]
            if not math.isfinite(b_val):
                continue  
            f0      = b_val - math.floor(b_val)
            if f0 < self.tol or f0 > 1.0 - self.tol:
                continue  

            entries     = []
            bound_width = {}
            other_bound = {}
            for k, nb_col in enumerate(nonbasic_cols):
                coef = Tsig[row_idx, k]
                if abs(coef) < self.tol:
                    continue

                kind       = 'int' if (nb_col < n and nb_col in bb._integer_vars) else 'cont'
                lower_case = not at_upper.get(nb_col, False)

                if nb_col >= n:
                    bound_val = 0.0
                    width     = math.inf if slack_sense[nb_col - n] != '=' else 0.0
                    oth_bound = math.inf if lower_case else -math.inf
                else:
                    bound_val = lb_full[nb_col] if lower_case else ub_full[nb_col]
                    width     = ub_full[nb_col] - lb_full[nb_col]
                    oth_bound = ub_full[nb_col] if lower_case else lb_full[nb_col]

                entries.append((nb_col, coef, kind, lower_case, bound_val))
                bound_width[nb_col] = width
                other_bound[nb_col] = oth_bound

            if not entries:
                continue

            rows.append(_CutRow(
                basic_col=col0, b_value=b_val, f0=f0, entries=entries,
                bound_width=bound_width, other_bound=other_bound,
                A_dense=A, b_vec=b_vec, n=n,
            ))

        return rows
 
    def _finalize_cut(self, entries_with_alpha, base_rhs, bb, sense, row: _CutRow,
                       extra_direct_terms: Optional[dict] = None):
        coeffs_x = {}
        rhs_x    = base_rhs
 
        if extra_direct_terms:
            for col, coef in extra_direct_terms.items():
                coeffs_x[col] = coeffs_x.get(col, 0.0) + coef
 
        for col, alpha, lower_case, bound_val in entries_with_alpha:
            if lower_case:
                coeffs_x[col] = coeffs_x.get(col, 0.0) + alpha
                rhs_x += alpha * bound_val
            else:
                coeffs_x[col] = coeffs_x.get(col, 0.0) - alpha
                rhs_x -= alpha * bound_val
 
        n = row.n
        slack_cols = [c for c in list(coeffs_x.keys()) if c >= n]
        for col in slack_cols:
            beta    = coeffs_x.pop(col)
            si      = col - n
            row_A   = row.A_dense[si, :]
            for k in range(n):
                akj = row_A[k]
                if abs(akj) > 1e-12:
                    coeffs_x[k] = coeffs_x.get(k, 0.0) - beta * akj
            rhs_x -= beta * row.b_vec[si]
 
        coeffs_x = {k: v for k, v in coeffs_x.items() if abs(v) > 1e-6}
        if not coeffs_x:
            return None
        return coeffs_x, rhs_x, sense
 
    def _add_cut_to_model(self, coeffs, rhs, sense, bb) -> bool:
        if not coeffs:
            return False
        vars_list = bb._base_model.getVars()
        expr = gp.quicksum(v * vars_list[k] for k, v in coeffs.items())
        name = f"_cut_{self.__class__.__name__}_{bb._base_model.NumConstrs}"
        if sense == '>=':
            bb._base_model.addConstr(expr >= rhs, name=name)
        else:
            bb._base_model.addConstr(expr <= rhs, name=name)
        bb._base_model.update()
        return True
 
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root_only={self.only_root})"
    
 
class MixedIntegerGomoryCut(CuttingPlane):
 
    def generate_cut(self, row: _CutRow, bb):
        f0 = row.f0
        entries_with_alpha = []
        for col, a, kind, lower_case, bound_val in row.entries:
            alpha = self._gmi_alpha(a, f0, kind)
            entries_with_alpha.append((col, alpha, lower_case, bound_val))
        return self._finalize_cut(entries_with_alpha, base_rhs=1.0, bb=bb, sense='>=', row=row)
 
    @staticmethod
    def _gmi_alpha(a: float, f0: float, kind: str) -> float:
        if kind == 'int':
            fj = a - math.floor(a)
            return fj / f0 if fj <= f0 else (1.0 - fj) / (1.0 - f0)
        else:
            return a / f0 if a >= 0 else -a / (1.0 - f0)
        
 
class SimpleMIR(CuttingPlane):

    def run(self, node, bb) -> int:
        if not self.should_run(node, bb):
            return 0
        if random.random() >= self.probability:
            return 0
        
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
        if model.Status != GRB.OPTIMAL:
            return 0

        constrs = model.getConstrs()
        if not constrs:
            return 0

        A = model.getA().toarray()
        n = bb._n
        added = 0

        for i, c in enumerate(constrs):
            if added >= self.max_cuts:
                break

            if c.Sense != '<':
                continue

            b = c.RHS
            f = b - math.floor(b)

            if f <= self.tol or f >= 1.0 - self.tol:
                continue

            coeffs = {}
            valid_row = True
            has_nonzero = False

            for j in range(n):
                coef = A[i, j]
                if abs(coef) <= self.tol:
                    continue

                has_nonzero = True
                v = vars_list[j]

                if v.LB < -self.tol:
                    valid_row = False
                    break

                if j in bb._integer_vars:
                    a_floor = math.floor(coef)
                    fj = coef - a_floor

                    if fj > f + self.tol:
                        a_hat = a_floor + (fj - f) / (1.0 - f)
                    else:
                        a_hat = a_floor

                    if abs(a_hat) > self.tol:
                        coeffs[j] = a_hat

                else:
                    if coef < -self.tol:
                        g_hat = coef / (1.0 - f)
                    else:
                        g_hat = 0.0

                    if abs(g_hat) > self.tol:
                        coeffs[j] = g_hat

            if not valid_row or not has_nonzero or not coeffs:
                continue

            rhs_cut = math.floor(b)

            lhs_lp = sum(coeffs[j] * vars_list[j].X for j in coeffs)

            if lhs_lp <= rhs_cut + self.tol:
                continue

            if self._add_cut_to_model(coeffs, rhs_cut, '<=', bb):
                added += 1

        return added

    def generate_cut(self, row: _CutRow, bb):
        return None
    
 
class LiftingExtending(CuttingPlane):
    def generate_cut(self, row: _CutRow, bb):
        f0 = row.f0
        entries_with_alpha = []
 
        for col, a, kind, lower_case, bound_val in row.entries:
            alpha_direct = MixedIntegerGomoryCut._gmi_alpha(a, f0, kind)
 
            width = row.bound_width.get(col, math.inf)
            if math.isfinite(width) and width > self.tol:   
                alpha_complement = MixedIntegerGomoryCut._gmi_alpha(-a, f0, kind)
                if alpha_complement > alpha_direct:
                    other_bound = row.other_bound.get(col, bound_val)
                    entries_with_alpha.append((col, alpha_complement, not lower_case, other_bound))
                    continue
 
            entries_with_alpha.append((col, alpha_direct, lower_case, bound_val))
 
        return self._finalize_cut(entries_with_alpha, base_rhs=1.0, bb=bb, sense='>=', row=row)
 

class CoverCut(CuttingPlane):

    def run(self, node, bb) -> int:
        if not self.should_run(node, bb):
            return 0
        if random.random() >= self.probability:
            return 0

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
        if model.Status != GRB.OPTIMAL:
            return 0

        constrs = model.getConstrs()
        if not constrs:
            return 0

        A = model.getA().toarray()
        n = bb._n
        added = 0

        for i, c in enumerate(constrs):
            if added >= self.max_cuts:
                break

            if c.Sense != '<':
                continue

            b = c.RHS
            if not np.isfinite(b):
                continue

            cols = []
            weights = []
            x_star = []
            valid_row = True

            for j in range(n):
                coef = A[i, j]
                if abs(coef) <= self.tol:
                    continue

                v = vars_list[j]

                if coef <= self.tol:
                    valid_row = False
                    break

                if bb._orig_vtype[j] != GRB.BINARY:
                    valid_row = False
                    break

                cols.append(j)
                weights.append(float(coef))
                x_star.append(float(v.X))

            if not valid_row or len(cols) < 2:
                continue

            if sum(weights) <= b + self.tol:
                continue

            result = self._find_minimal_cover(weights, b, x_star)
            if result is None:
                continue

            cover_local, violation = result
            if violation <= self.tol:
                continue

            coeffs = {cols[k]: 1.0 for k in cover_local}
            rhs_cut = float(len(cover_local) - 1)

            if self._add_cut_to_model(coeffs, rhs_cut, '<=', bb):
                added += 1

        return added

    def generate_cut(self, row: _CutRow, bb):
        return None

    def _find_minimal_cover(self, a, b, x_star, eps=1e-6):
        
        a = np.asarray(a, dtype=float)
        x_star = np.asarray(x_star, dtype=float)
        n = len(a)

        if n == 0:
            return None

        c = 1.0 - x_star

        A_ub = np.array([-a], dtype=float)
        b_ub = np.array([-(b + eps)], dtype=float)

        try:
            from scipy.optimize import linprog
        except ImportError:
            return None

        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(0.0, 1.0)] * n,
            method="highs",
        )

        if not res.success:
            return None

        order = sorted(range(n), key=lambda j: (-res.x[j], c[j]))

        C = []
        total = 0.0
        for j in order:
            C.append(j)
            total += a[j]
            if total > b + eps:
                break

        if total <= b + eps:
            return None  
        
        changed = True
        while changed:
            changed = False
            for j in sorted(C, key=lambda j: -a[j]):
                if total - a[j] > b + eps:
                    C.remove(j)
                    total -= a[j]
                    changed = True
                    break

        violation = float(np.sum(x_star[C]) - (len(C) - 1))
        return C, violation
