# vi_mdp.py
# Value Iteration for a Grid Maze (MDP) — polished & connected, step-cost fixed

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

Coord = Tuple[int, int]  # (row, col)

# -----------------------------
# CONFIG AREA — EDIT TO YOUR MAZE
# -----------------------------
H, W = 7, 7  # grid height, width

# 7x7 maze (S at (0,0), G at (6,6)); '#' are walls below
# Row 0: S . # . # . .
# Row 1: . . . . . . .   <-- (1,2) opened to make maze connected
# Row 2: # # # . # # .
# Row 3: . . . . . . .
# Row 4: . # . # . # .
# Row 5: . # . # . # .
# Row 6: . . . # . . G

WALLS = {
    (0,2), (0,4),
    # (1,2) REMOVED to connect start region to corridor
    (2,0), (2,1), (2,2), (2,4), (2,5),
    # row 3 has no walls
    (4,1), (4,3), (4,5),
    (5,1), (5,3), (5,5),
    (6,3),
}

START: Coord = (0, 0)
TERMINALS: Dict[Coord, float] = {(6, 6): 0.0}  # terminal payoff 0

# Actions: 0:up, 1:down, 2:left, 3:right
ACTIONS: List[Coord] = [(-1,0), (1,0), (0,-1), (0,1)]

# Dynamics / rewards
GAMMA = 0.99        # prefer shortest path
THETA = 1e-6        # stopping tolerance
STEP_REWARD = -1.0  # −1 per time-step
SLIP_P = 0.0        # deterministic

# -----------------------------
# MDP Implementation
# -----------------------------

def lateral_indices(a_idx: int) -> Tuple[int, int]:
    # For up/down: lateral are left/right; for left/right: lateral are up/down
    return (2, 3) if a_idx in (0, 1) else (0, 1)

@dataclass
class GridMDP:
    height: int
    width: int
    walls: set
    terminals: Dict[Coord, float]
    actions: List[Coord]
    gamma: float
    step_reward: float
    slip_p: float

    def __post_init__(self):
        self.states: List[Coord] = [
            (r, c) for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in self.walls
        ]
        self.idx: Dict[Coord, int] = {s: i for i, s in enumerate(self.states)}
        self.is_terminal_mask = np.array(
            [s in self.terminals for s in self.states], dtype=bool
        )

    def in_grid(self, rc: Coord) -> bool:
        r, c = rc
        return 0 <= r < self.height and 0 <= c < self.width and rc not in self.walls

    def step_transitions(self, s: Coord, a_idx: int) -> List[Tuple[float, Coord, float]]:
        """Return list of (prob, next_state, reward) for taking action a_idx in s."""
        # Terminal is absorbing with its terminal payoff only (no step cost after termination)
        if s in self.terminals:
            r_term = self.terminals[s]
            return [(1.0, s, r_term)]

        r, c = s
        intended = self.actions[a_idx]
        li, ri = lateral_indices(a_idx)

        candidates: List[Tuple[float, Coord]] = []
        p_intended = max(0.0, 1.0 - 2.0 * self.slip_p)
        if p_intended > 0:
            candidates.append((p_intended, intended))
        if self.slip_p > 0:
            candidates.append((self.slip_p, self.actions[li]))
            candidates.append((self.slip_p, self.actions[ri]))

        out: List[Tuple[float, Coord, float]] = []
        for p, (dr, dc) in candidates:
            nr, nc = r + dr, c + dc
            ns = (nr, nc) if self.in_grid((nr, nc)) else s  # bounce if hitting wall/border
            # FIX: pay step cost every move; add terminal payoff if we land in terminal
            base = self.step_reward
            bonus = self.terminals.get(ns, 0.0)
            rew = base + bonus
            out.append((p, ns, rew))
        return out

    def q_value(self, s: Coord, a_idx: int, V: np.ndarray) -> float:
        return sum(
            p * (rew + self.gamma * V[self.idx[ns]])
            for p, ns, rew in self.step_transitions(s, a_idx)
        )

    def value_iteration(self, theta: float = 1e-6, max_iters: int = 10_000) -> Tuple[np.ndarray, int]:
        V = np.zeros(len(self.states), dtype=float)
        # Fix terminal values to their terminal payoff (absorbing)
        for s, r in self.terminals.items():
            V[self.idx[s]] = r

        for it in range(1, max_iters + 1):
            delta = 0.0
            V_new = V.copy()
            for s in self.states:
                i = self.idx[s]
                if self.is_terminal_mask[i]:
                    continue  # keep terminal fixed
                V_new[i] = max(self.q_value(s, a, V) for a in range(len(self.actions)))
                delta = max(delta, abs(V_new[i] - V[i]))
            V = V_new
            if delta < theta:
                return V, it
        return V, max_iters  # fallback

    def extract_policy(self, V: np.ndarray) -> Dict[Coord, str]:
        arrow = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        policy: Dict[Coord, str] = {}
        for s in self.states:
            if s in self.terminals:
                policy[s] = 'T'
            else:
                q_vals = [self.q_value(s, a, V) for a in range(len(self.actions))]
                policy[s] = arrow[int(np.argmax(q_vals))]
        return policy

    def pretty_print(self, V: np.ndarray, policy: Dict[Coord, str], start: Coord | None = None) -> None:
        print("\n=== Value Table & Policy Map ===")
        for r in range(self.height):
            row_val, row_pi = [], []
            for c in range(self.width):
                cell = (r, c)
                if cell in self.walls:
                    row_val.append(" #### ")
                    row_pi.append(" ## ")
                else:
                    v = V[self.idx[cell]]
                    row_val.append(f"{v:6.2f}")
                    if cell in self.terminals:
                        row_pi.append("  T  ")
                    elif start is not None and cell == start:
                        row_pi.append("  S  ")
                    else:
                        row_pi.append(f"  {policy[cell]}  ")
            print(" ".join(row_val), "   ", " ".join(row_pi))
        print()

def main():
    mdp = GridMDP(
        height=H, width=W, walls=WALLS, terminals=TERMINALS,
        actions=ACTIONS, gamma=GAMMA, step_reward=STEP_REWARD, slip_p=SLIP_P
    )
    V, iters = mdp.value_iteration(theta=THETA)
    policy = mdp.extract_policy(V)
    print(f"Converged in {iters} iteration(s) with theta={THETA}")
    mdp.pretty_print(V, policy, start=START)

if __name__ == "__main__":
    main()
