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

class BranchDirection(Enum):
    CEIL_FIRST  = "ceil_first"
    FLOOR_FIRST = "floor_first"

class BranchVariableRule(ABC):
    @abstractmethod
    def choose_variable(self, node: "Node", bb: "BranchAndBound") -> Optional[int]:
        pass

    def on_node_solved(self, node: "Node", bb: "BranchAndBound") -> None:
        pass

    def on_branch_result(
        self,
        parent: "Node",
        var_idx: int,
        child_floor: Optional["Node"],
        floor_status: Optional[str],
        child_ceil: Optional["Node"],
        ceil_status: Optional[str],
        bb: "BranchAndBound",
    ) -> None:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__

class FirstFractionalRule(BranchVariableRule):
    def choose_variable(self, node: "Node", bb: "BranchAndBound") -> Optional[int]:
        for j in bb._integer_vars:
            x = node.lp_solution[j]
            if not bb._is_integral_value(x):
                return j
        return None

class MostFractionalRule(BranchVariableRule):
    def choose_variable(self, node: "Node", bb: "BranchAndBound") -> Optional[int]:
        best_j = None
        best_score = -1.0

        for j in bb._integer_vars:
            x = node.lp_solution[j]
            frac = x - math.floor(x)
            if frac <= bb.tol or frac >= 1.0 - bb.tol:
                continue

            score = min(frac, 1.0 - frac)
            if score > best_score:
                best_score = score
                best_j = j

        return best_j

class StrongBranchingRule(BranchVariableRule):
    def __init__(
        self,
        max_candidates: int = 10,
        candidate_rule: Optional[BranchVariableRule] = None,
        score_mode: str = "product",
    ):
        self.max_candidates = max_candidates
        self.candidate_rule = candidate_rule or MostFractionalRule()
        self.score_mode = score_mode

    def choose_variable(self, node: "Node", bb: "BranchAndBound") -> Optional[int]:
        self.last_candidates = []
        self.last_selected = None
        candidates = self._get_candidates(node, bb)
        if not candidates:
            return None

        parent_bound = node.lp_bound
        best_j = None
        best_score = -float("inf")

        for j in candidates:
            floor_status, floor_bound = bb._evaluate_branch_side(node, j, "floor")
            ceil_status, ceil_bound = bb._evaluate_branch_side(node, j, "ceil")

            gain_floor = bb._branch_gain(parent_bound, floor_status, floor_bound)
            gain_ceil = bb._branch_gain(parent_bound, ceil_status, ceil_bound)

            score = self._score(gain_floor, gain_ceil)

            self.last_candidates.append({
                "var_idx": j,
                "gain_floor": gain_floor,
                "gain_ceil": gain_ceil,
                "score": score
            })

            if score > best_score:
                best_score = score
                best_j = j
        self.last_selected = best_j
        return best_j

    def _get_candidates(self, node: "Node", bb: "BranchAndBound") -> list[int]:
        frac_vars = []
        for j in bb._integer_vars:
            x = node.lp_solution[j]
            frac = x - math.floor(x)
            if frac <= bb.tol or frac >= 1.0 - bb.tol:
                continue
            score = min(frac, 1.0 - frac)
            frac_vars.append((score, j))

        frac_vars.sort(reverse=True) 
        return [j for _, j in frac_vars[: self.max_candidates]]

    def _score(self, gain_floor: float, gain_ceil: float) -> float:
        if self.score_mode == "min":
            return min(gain_floor, gain_ceil)
        elif self.score_mode == "sum":
            return gain_floor + gain_ceil
        else:  
            return max(0.0, gain_floor) * max(0.0, gain_ceil)

class PseudoCostRule(BranchVariableRule):
    def __init__(
        self,
        reliability: int = 2,
        max_strong_candidates: int = 6,
        score_mode: str = "product",
    ):
        self.reliability = reliability
        self.max_strong_candidates = max_strong_candidates
        self.score_mode = score_mode

        self.down_sum = {}
        self.down_count = {}
        self.up_sum = {}
        self.up_count = {}

    def choose_variable(self, node: "Node", bb: "BranchAndBound") -> Optional[int]:
        self.last_candidates = []
        self.last_selected = None

        candidates = []
        unreliable = []

        for j in bb._integer_vars:
            x = node.lp_solution[j]
            frac = x - math.floor(x)
            if frac <= bb.tol or frac >= 1.0 - bb.tol:
                continue

            if self._is_reliable(j):
                score = self._estimated_score(j, x)
                candidates.append((score, j))

                self.last_candidates.append({
                    "var_idx": j,
                    "var_name": bb._names[j],
                    "mode": "pseudocost",
                    "lp_value": x,
                    "fractionality": min(frac, 1.0 - frac),
                    "score": score,
                    "pc_down": self._pc_down(j),
                    "pc_up": self._pc_up(j),
                    "reliable_down": self.down_count.get(j, 0),
                    "reliable_up": self.up_count.get(j, 0),
                })
            else:
                unreliable.append(j)

        if candidates:
            candidates.sort(reverse=True)
            self.last_selected = candidates[0][1]
            return candidates[0][1]

        if unreliable:
            ranked = sorted(
                unreliable,
                key=lambda j: min(
                    node.lp_solution[j] - math.floor(node.lp_solution[j]),
                    math.ceil(node.lp_solution[j]) - node.lp_solution[j],
                ),
                reverse=True,
            )
            best_j = None
            best_score = -float("inf")
            parent_bound = node.lp_bound

            for j in ranked[: self.max_strong_candidates]:
                floor_status, floor_bound = bb._evaluate_branch_side(node, j, "floor")
                ceil_status, ceil_bound = bb._evaluate_branch_side(node, j, "ceil")

                gain_floor = bb._branch_gain(parent_bound, floor_status, floor_bound)
                gain_ceil = bb._branch_gain(parent_bound, ceil_status, ceil_bound)
                score = self._combine(gain_floor, gain_ceil)

                x = node.lp_solution[j]
                frac = x - math.floor(x)

                self.last_candidates.append({
                    "var_idx": j,
                    "var_name": bb._names[j],
                    "mode": "strong_branching_fallback",
                    "lp_value": x,
                    "fractionality": min(frac, 1.0 - frac),
                    "score": score,
                    "gain_floor": gain_floor,
                    "gain_ceil": gain_ceil,
                    "floor_status": floor_status,
                    "ceil_status": ceil_status,
                    "reliable_down": self.down_count.get(j, 0),
                    "reliable_up": self.up_count.get(j, 0),
                })

                if score > best_score:
                    best_score = score
                    best_j = j

            self.last_selected = best_j
            return best_j

        return None

    def on_branch_result(
        self,
        parent: "Node",
        var_idx: int,
        child_floor: Optional["Node"],
        floor_status: Optional[str],
        child_ceil: Optional["Node"],
        ceil_status: Optional[str],
        bb: "BranchAndBound",
    ) -> None:
        parent_bound = parent.lp_bound
        x = parent.lp_solution[var_idx]

        delta_down = x - math.floor(x)
        delta_up = math.ceil(x) - x

        floor_bound = child_floor.lp_bound if (child_floor is not None and floor_status == "optimal") else None
        gain_floor = bb._branch_gain(parent_bound, floor_status, floor_bound)
        if delta_down > bb.tol:
            pc_down = gain_floor / delta_down
            self.down_sum[var_idx] = self.down_sum.get(var_idx, 0.0) + pc_down
            self.down_count[var_idx] = self.down_count.get(var_idx, 0) + 1

        ceil_bound = child_ceil.lp_bound if (child_ceil is not None and ceil_status == "optimal") else None
        gain_ceil = bb._branch_gain(parent_bound, ceil_status, ceil_bound)
        if delta_up > bb.tol:
            pc_up = gain_ceil / delta_up
            self.up_sum[var_idx] = self.up_sum.get(var_idx, 0.0) + pc_up
            self.up_count[var_idx] = self.up_count.get(var_idx, 0) + 1

    def _is_reliable(self, j: int) -> bool:
        return (
            self.down_count.get(j, 0) >= self.reliability
            and self.up_count.get(j, 0) >= self.reliability
        )

    def _pc_down(self, j: int) -> float:
        cnt = self.down_count.get(j, 0)
        if cnt == 0:
            return 0.0
        return self.down_sum[j] / cnt

    def _pc_up(self, j: int) -> float:
        cnt = self.up_count.get(j, 0)
        if cnt == 0:
            return 0.0
        return self.up_sum[j] / cnt

    def _estimated_score(self, j: int, x: float) -> float:
        delta_down = x - math.floor(x)
        delta_up = math.ceil(x) - x
        gain_down = self._pc_down(j) * delta_down
        gain_up = self._pc_up(j) * delta_up
        return self._combine(gain_down, gain_up)

    def _combine(self, gain_down: float, gain_up: float) -> float:
        if self.score_mode == "min":
            return min(gain_down, gain_up)
        elif self.score_mode == "sum":
            return gain_down + gain_up
        else:
            return max(0.0, gain_down) * max(0.0, gain_up)
