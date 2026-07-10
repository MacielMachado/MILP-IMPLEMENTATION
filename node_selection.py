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

from branching import *


@dataclass
class Node:
    node_id: int = -1
    parent_id: Optional[int] = None
    lower_bounds: dict = field(default_factory=dict)
    upper_bounds: dict = field(default_factory=dict)
    depth: int = 0
    lp_bound: float = float("inf")
    lp_solution: dict = field(default_factory=dict)

    parent_branch_var: Optional[int] = None
    parent_branch_side: Optional[str] = None
    parent_lp_bound: Optional[float] = None
    parent_lp_value: Optional[float] = None

    def is_integer_feasible(self, integer_vars: set, tol: float = 1e-6) -> bool:
        return all(
            abs(self.lp_solution[i] - round(self.lp_solution[i])) <= tol
            for i in integer_vars
            if i in self.lp_solution
        )

    def first_fractional(self, integer_vars: set, tol: float = 1e-6) -> Optional[int]:
        for idx in integer_vars:
            if idx not in self.lp_solution:
                continue
            if abs(self.lp_solution[idx] - round(self.lp_solution[idx])) > tol:
                return idx
        return None

class NodeSelection(ABC):

    @abstractmethod
    def add_children(self, open_nodes, child_ceil: Node, child_floor: Node, direction: BranchDirection) -> None:
        pass

    @abstractmethod
    def add_node(self, open_nodes, node: Node) -> None:
        pass

    @abstractmethod
    def select_node(self, open_nodes, sense) -> Node:
        pass

    @abstractmethod
    def initialize(self) -> object:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


class DepthFirst(NodeSelection):
    def add_children(self, open_nodes, child_ceil, child_floor, direction):
        if direction == BranchDirection.CEIL_FIRST:
            open_nodes.append(child_floor)
            open_nodes.append(child_ceil)
        else:
            open_nodes.append(child_ceil)
            open_nodes.append(child_floor)

    def add_node(self, open_nodes, node):
        open_nodes.append(node)

    def initialize(self):
        return []

    def select_node(self, open_nodes, sense):
        return open_nodes.pop()


class BestFirst(NodeSelection):
    def add_children(self, open_nodes, child_ceil, child_floor, direction):
        if direction == BranchDirection.CEIL_FIRST:
            open_nodes.append(child_floor)
            open_nodes.append(child_ceil)
        else:
            open_nodes.append(child_ceil)
            open_nodes.append(child_floor)

    def add_node(self, open_nodes, node):
        open_nodes.append(node)

    def initialize(self):
        return []

    def select_node(self, open_nodes, sense):
        if sense == GRB.MAXIMIZE:
            best_idx = max(range(len(open_nodes)), key=lambda i: open_nodes[i].lp_bound)
        else:
            best_idx = min(range(len(open_nodes)), key=lambda i: open_nodes[i].lp_bound)
        open_nodes[best_idx], open_nodes[-1] = open_nodes[-1], open_nodes[best_idx]
        return open_nodes.pop()


class BreadthFirst(NodeSelection):
    def add_children(self, open_nodes, child_ceil, child_floor, direction):
        if direction == BranchDirection.CEIL_FIRST:
            open_nodes.append(child_ceil)
            open_nodes.append(child_floor)
        else:
            open_nodes.append(child_floor)
            open_nodes.append(child_ceil)

    def add_node(self, open_nodes, node):
        open_nodes.append(node)

    def initialize(self):
        return deque()

    def select_node(self, open_nodes, sense):
        return open_nodes.popleft()
