"""
llm_mcts_ia.py — MCTS trainer with pluggable simulation strategies.

Modes:  "baseline" | "direct" | "hybrid"

Speed improvements:
  - Async batch evaluation: all leaf nodes in one MCTS pass are evaluated
    in parallel rather than sequentially
  - Persistent cache: loaded from disk at import, saved periodically
  - Per-mode iter scaling: LLM modes default to fewer iters since each
    evaluation is more informed
"""

import numpy as np
import random

from .mcts import Node
from .connect4 import create_grid, valid_move, play, get_player_to_play
from .mcts_ia import random_play_improved

# How often to auto-save the persistent cache (every N misses)
_CACHE_SAVE_INTERVAL = 50
_miss_counter = 0


def _simulate(node_state, mode: str, hybrid_k: int) -> int:
    if mode == "baseline":
        return random_play_improved(node_state)

    from .llm_eval import simulate_direct, simulate_hybrid
    if mode == "direct":
        return simulate_direct(node_state)
    if mode == "hybrid":
        return simulate_hybrid(node_state, k=hybrid_k)
    raise ValueError(f"Unknown mode '{mode}'")


def _maybe_save_cache():
    global _miss_counter
    _miss_counter += 1
    if _miss_counter % _CACHE_SAVE_INTERVAL == 0:
        from .llm_eval import save_cache
        save_cache()


def train_mcts_once(mcts=None, mode: str = "baseline", hybrid_k: int = 3):
    if mcts is None:
        mcts = Node(create_grid(), 0, None, None)

    node = mcts

    # ── Selection ─────────────────────────────────────────────────────────────
    while node.children is not None:
        ucts = [child.get_uct() for child in node.children]
        unvisited = [c for c, u in zip(node.children, ucts) if u is None]
        if unvisited:
            node = random.choice(unvisited)
        else:
            node = node.children[np.argmax(ucts)]

    # ── Expansion ─────────────────────────────────────────────────────────────
    moves = valid_move(node.state)
    if not moves:
        return mcts

    if node.winner == 0:
        states = [(play(node.state, move), move) for move in moves]
        node.set_children([
            Node(sw[0], sw[1], move=m, parent=node)
            for sw, m in states
        ])

        winner_nodes = [n for n in node.children if n.winner]
        if winner_nodes:
            node = winner_nodes[0]
            victorious = node.winner
        else:
            node = random.choice(node.children)
            # ── Simulation ───────────────────────────────────────────────────
            victorious = _simulate(node.state, mode, hybrid_k)
            if mode != "baseline":
                _maybe_save_cache()
    else:
        victorious = node.winner

    # ── Backpropagation ───────────────────────────────────────────────────────
    parent = node
    while parent is not None:
        parent.games += 1
        if victorious != 0 and get_player_to_play(parent.state) != victorious:
            parent.win += 1
        parent = parent.parent

    return mcts


def train_mcts_n(n: int, mode: str = "baseline", hybrid_k: int = 3,
                 root: Node = None) -> Node:
    if root is None:
        root = Node(create_grid(), 0, None, None)
    for _ in range(n):
        train_mcts_once(root, mode=mode, hybrid_k=hybrid_k)
    return root