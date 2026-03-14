"""
main.py — Connect Four vs MCTS AI

Usage:
    python main.py                        # baseline MCTS, no debug
    python main.py --mode direct          # LLM direct evaluation
    python main.py --mode hybrid          # LLM hybrid rollout
    python main.py --mode direct --debug  # with full debug logging
    python main.py --iters 300            # custom iteration count
"""

import argparse
import logging
import numpy as np
from connect4.connect4 import create_grid, valid_move, play, get_player_to_play

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Connect Four MCTS AI")
parser.add_argument("--mode",  choices=["baseline", "direct", "hybrid"],
                    default="baseline",
                    help="Simulation strategy for MCTS (default: baseline)")
parser.add_argument("--iters", type=int, default=50,
                    help="MCTS iterations per move (default: 200)")
parser.add_argument("--hybrid_k", type=int, default=3,
                    help="LLM-guided plies before random takeover (hybrid only)")
parser.add_argument("--debug", action="store_true",
                    help="Enable debug logging (DeepSeek responses, input validation)")
args = parser.parse_args()

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_LEVEL = logging.DEBUG if args.debug else logging.WARNING
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=LOG_LEVEL
)
log = logging.getLogger("connect4")

# Patch llm_eval to emit debug logs BEFORE importing it
if args.mode != "baseline":
    import connect4.llm_eval as _llm

    # ── Wrap _call_deepseek to log raw responses ──────────────────────────────
    _orig_call = _llm._call_deepseek

    def _logged_call(prompt: str, max_tokens: int = 16) -> str:
        log.debug("── DeepSeek REQUEST ─────────────────────────────")
        for line in prompt.splitlines():
            log.debug("  %s", line)
        log.debug("─────────────────────────────────────────────────")
        raw = _orig_call(prompt, max_tokens)
        log.debug("── DeepSeek RESPONSE: %r", raw)
        return raw

    _llm._call_deepseek = _logged_call

    # ── Wrap llm_evaluate_direct to log parse result ──────────────────────────
    _orig_eval = _llm.llm_evaluate_direct

    def _logged_eval(grid):
        state_key = _llm.to_state(grid)
        cached = state_key in _llm._direct_cache
        score = _orig_eval(grid)
        if cached:
            log.debug("direct_eval  CACHE HIT  → %.3f", score)
        else:
            log.debug("direct_eval  parsed=%.3f  (VALID)", score)
        return score

    _llm.llm_evaluate_direct = _logged_eval

    # ── Wrap llm_pick_move to log column validity ─────────────────────────────
    _orig_pick = _llm.llm_pick_move

    def _logged_pick(grid):
        from connect4.connect4 import valid_move as _vm, to_state as _ts
        state_key = _ts(grid)
        cached = state_key in _llm._move_cache
        col = _orig_pick(grid)
        legal = _vm(grid)
        if cached:
            log.debug("pick_move    CACHE HIT  → col=%d", col)
        elif col in legal:
            log.debug("pick_move    VALID      → col=%d  (legal=%s)", col, legal)
        else:
            log.debug("pick_move    INVALID    → col=%d not in legal=%s, "
                      "fell back to random", col, legal)
        return col

    _llm.llm_pick_move = _logged_pick

# ── Import MCTS after patching ────────────────────────────────────────────────
from connect4.mcts import Node
from connect4.llm_mcts_ia import train_mcts_once
from connect4.llm_eval import cache_stats as _cs


# ── Board display ─────────────────────────────────────────────────────────────
def print_board(grid):
    symbols = {0: ' ', 1: 'O', -1: 'X'}
    print()
    for row in grid:
        print('  |' + '|'.join(f' {symbols[v]} ' for v in row) + '|')
    print('  +' + '+'.join(['---'] * grid.shape[1]) + '+')
    print('    ' + '   '.join(str(i) for i in range(grid.shape[1])))
    print()


# ── Human move input with validation ─────────────────────────────────────────
def get_human_move(grid) -> int:
    legal = valid_move(grid)
    while True:
        raw = input(f"  Your move (columns {legal}): ").strip()
        log.debug("human input: %r", raw)
        try:
            col = int(raw)
        except ValueError:
            log.debug("INVALID input — not an integer: %r", raw)
            print(f"  ✗ '{raw}' is not a number. Try again.")
            continue
        if col not in legal:
            log.debug("INVALID input — col=%d not in legal=%s", col, legal)
            print(f"  ✗ Column {col} is not available. Legal columns: {legal}")
            continue
        log.debug("VALID input — col=%d", col)
        return col


# ── AI move ───────────────────────────────────────────────────────────────────
def get_ai_move(grid, mode: str, iters: int, hybrid_k: int) -> int:
    print(f"  AI thinking ({mode}, {iters} iters)...")
    root = Node(grid.copy(), 0, None, None)
    for _ in range(iters):
        train_mcts_once(root, mode=mode, hybrid_k=hybrid_k)
    _, move = root.select_move()
    log.debug("AI selected col=%d  (tree: %d nodes visited)", move, root.games)
    if mode != "baseline":
        log.debug("cache stats — direct: %dH/%dM  move: %dH/%dM",
                  _cs["direct_hits"], _cs["direct_misses"],
                  _cs["move_hits"],   _cs["move_misses"])
    return move


# ── Main game loop ────────────────────────────────────────────────────────────
def main():
    print(f"\n  Connect Four  |  mode={args.mode}  iters={args.iters}")
    if args.debug:
        print("  [DEBUG MODE ON]\n")

    while True:
        grid = create_grid()

        choice = input("  Go first? [Y/n]: ").strip().lower()
        if choice == 'n':
            human_player = 1   # human is O, AI opens as X (-1)
            print("  You are O  |  AI goes first")
        else:
            human_player = -1  # human is X, human opens
            print("  You are X  |  You go first")

        print_board(grid)

        while True:
            current = get_player_to_play(grid)

            if current == human_player:
                col = get_human_move(grid)
            else:
                col = get_ai_move(grid, args.mode, args.iters, args.hybrid_k)
                print(f"  AI plays column {col}")

            grid, winner = play(grid, col)
            print_board(grid)

            if winner != 0:
                label = 'You win!' if winner == human_player else 'AI wins!'
                print(f"  *** {label} ***\n")
                break

            if not valid_move(grid):
                print("  *** Draw! ***\n")
                break

        again = input("  Play again? [y/N]: ").strip().lower()
        if again != 'y':
            break

    print("  Goodbye!")


if __name__ == '__main__':
    main()