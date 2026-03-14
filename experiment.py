"""
experiment.py — Part 3 experiment runner.

Metrics logged per move:
  - win_rate_chosen : win rate of the chosen move node (exploitation signal)
  - avg_tree_depth  : average depth reached across all MCTS iterations
  - move_time_s     : wall-clock seconds spent on this move

Usage:
    python experiment.py --games 10 --iters 50 200
    python experiment.py --games 20 --iters 50 200 500 --llm_iters_scale 1.0
"""

import argparse
import csv
import json
import logging
# Suppress OpenAI/httpx HTTP logs before client is ever imported
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
import os
import random
import time
from datetime import datetime

import numpy as np

from connect4.connect4 import create_grid, valid_move, play, get_player_to_play
from connect4.mcts import Node
from connect4.llm_mcts_ia import train_mcts_once
from connect4.llm_eval import cache_stats

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Connect Four MCTS experiment runner")
parser.add_argument("--games",           type=int,   default=10)
parser.add_argument("--iters",           type=int,   nargs="+", default=[50, 200, 500])
parser.add_argument("--hybrid_k",        type=int,   default=3)
parser.add_argument("--llm_iters_scale", type=float, default=0.5,
                    help="LLM modes use this fraction of --iters (default: 0.5)")
parser.add_argument("--logdir",          type=str,   default="results")
args = parser.parse_args()

# ── Log file setup ────────────────────────────────────────────────────────────
os.makedirs(args.logdir, exist_ok=True)
run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
log_txt  = os.path.join(args.logdir, f"run_{run_id}.log")
log_csv  = os.path.join(args.logdir, f"run_{run_id}_moves.csv")   # per-move
log_json = os.path.join(args.logdir, f"run_{run_id}_summary.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_txt), logging.StreamHandler()],
)
log = logging.getLogger("experiment")

# CSV 1 — one row per MOVE
csv_file   = open(log_csv, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "run_id", "matchup", "iters", "game_num", "move_num",
    "mode_moving", "chosen_col",
    "win_rate_chosen",
    "avg_tree_depth",
    "move_time_s",
    "cache_direct_hits", "cache_direct_misses",
    "cache_move_hits",   "cache_move_misses",
])

# CSV 2 — one row per GAME (win/draw/loss rates)
log_csv_games    = os.path.join(args.logdir, f"run_{run_id}_games.csv")
csv_games_file   = open(log_csv_games, "w", newline="")
csv_games_writer = csv.writer(csv_games_file)
csv_games_writer.writerow([
    "run_id", "matchup", "iters", "game_num",
    "mode_a", "mode_b",
    "mode_first", "mode_second",
    "outcome",
    "winner_mode",
    "num_moves", "game_time_s", "avg_move_time_s",
    "mode_a_wins", "mode_b_wins", "draws",
    "mode_a_winrate", "mode_b_winrate", "draw_rate",
])

log.info("=" * 60)
log.info("Experiment started  run_id=%s", run_id)
log.info("games=%d  iters=%s  hybrid_k=%d  llm_scale=%.2f",
         args.games, args.iters, args.hybrid_k, args.llm_iters_scale)
log.info("=" * 60)


# ── Tree metric helpers ───────────────────────────────────────────────────────

def get_win_rate_chosen(root: Node, chosen_col: int) -> float:
    """Win rate (win/games) of the child node corresponding to chosen_col."""
    if root.children is None:
        return 0.0
    for child in root.children:
        if child.move == chosen_col:
            return child.win / child.games if child.games > 0 else 0.0
    return 0.0


def get_avg_tree_depth(root: Node) -> float:
    """
    Average depth of all visited nodes in the tree.
    Depth 0 = root, depth 1 = root's children, etc.
    """
    total_depth = 0
    count       = 0

    def _walk(node: Node, depth: int):
        nonlocal total_depth, count
        if node.games > 0:
            total_depth += depth
            count       += 1
        if node.children:
            for child in node.children:
                _walk(child, depth + 1)

    _walk(root, 0)
    return round(total_depth / count, 2) if count > 0 else 0.0


# ── Core game logic ───────────────────────────────────────────────────────────

def play_game(mode_a: str, mode_b: str, iters: int,
              matchup: str, game_num: int,
              hybrid_k: int = 3, seed: int = None,
              llm_iters_scale: float = 0.5):
    """
    Play one full game, logging per-move metrics to CSV.
    Returns (winner_player: int, num_moves: int, avg_move_time: float)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    grid      = create_grid()
    modes     = {1: mode_a, -1: mode_b}
    move_num  = 0
    move_times = []

    while True:
        player = get_player_to_play(grid)
        mode   = modes[player]

        effective_iters = (int(iters * llm_iters_scale)
                           if mode != "baseline" else iters)
        effective_iters = max(10, effective_iters)

        # ── Build MCTS tree, timed ────────────────────────────────────────────
        m_start = time.time()
        root    = Node(grid.copy(), 0, None, None)
        for _ in range(effective_iters):
            train_mcts_once(root, mode=mode, hybrid_k=hybrid_k)
        m_elapsed = round(time.time() - m_start, 3)
        move_times.append(m_elapsed)

        # ── Select move ───────────────────────────────────────────────────────
        _, chosen_col = root.select_move()
        if chosen_col is None:
            return 0, move_num, float(np.mean(move_times)) if move_times else 0.0

        # ── Collect metrics ───────────────────────────────────────────────────
        win_rate  = get_win_rate_chosen(root, chosen_col)
        avg_depth = get_avg_tree_depth(root)

        move_num += 1

        # Write one CSV row per move
        csv_writer.writerow([
            run_id, matchup, iters, game_num, move_num,
            mode, chosen_col,
            round(win_rate, 4),
            avg_depth,
            m_elapsed,
            cache_stats["direct_hits"],  cache_stats["direct_misses"],
            cache_stats["move_hits"],    cache_stats["move_misses"],
        ])
        csv_file.flush()

        # ── Apply move ────────────────────────────────────────────────────────
        grid, winner = play(grid, chosen_col)

        if abs(winner) > 0:
            avg_t = round(float(np.mean(move_times)), 3)
            return player, move_num, avg_t

        if not valid_move(grid):
            avg_t = round(float(np.mean(move_times)), 3)
            return 0, move_num, avg_t


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(mode_a: str, mode_b: str, iters_list: list,
                   num_games: int, hybrid_k: int = 3,
                   llm_iters_scale: float = 0.5) -> dict:
    matchup = f"{mode_a}_vs_{mode_b}"
    log.info("")
    log.info("━" * 60)
    log.info("MATCHUP: %s", matchup.upper())
    log.info("━" * 60)

    results = {}

    for iters in iters_list:
        wins_a, wins_b, draws = 0, 0, 0
        all_move_times = []
        t_start = time.time()

        log.info("  ── iters=%d ──", iters)

        for g in range(num_games):
            if g % 2 == 0:
                first, second, flip = mode_a, mode_b, False
            else:
                first, second, flip = mode_b, mode_a, True

            g_start = time.time()
            outcome_player, num_moves, avg_move_t = play_game(
                first, second, iters,
                matchup=matchup, game_num=g + 1,
                hybrid_k=hybrid_k, seed=g,
                llm_iters_scale=llm_iters_scale,
            )
            g_elapsed = round(time.time() - g_start, 1)
            all_move_times.append(avg_move_t)

            if outcome_player == 0:
                outcome_label = "draw";  winner_mode = "draw"; draws += 1
            elif (outcome_player == 1 and not flip) or (outcome_player == -1 and flip):
                outcome_label = "A";  winner_mode = mode_a;  wins_a += 1
            else:
                outcome_label = "B";  winner_mode = mode_b;  wins_b += 1

            log.info(
                "    game %2d/%d  first=%-8s  winner=%-8s  "
                "moves=%2d  avg_move=%.2fs  total=%.1fs",
                g + 1, num_games, first, winner_mode,
                num_moves, avg_move_t, g_elapsed,
            )

            # Write per-game CSV row with cumulative W/D/L rates
            total_games = g + 1
            csv_games_writer.writerow([
                run_id, matchup, iters, total_games,
                mode_a, mode_b,
                first, second,
                outcome_label,
                winner_mode,
                num_moves, round(g_elapsed, 2), avg_move_t,
                wins_a, wins_b, draws,
                round(wins_a / total_games, 4),
                round(wins_b / total_games, 4),
                round(draws  / total_games, 4),
            ])
            csv_games_file.flush()

        elapsed = time.time() - t_start
        results[iters] = {
            "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
            "time_s": round(elapsed, 1),
            "avg_move_time_s": round(float(np.mean(all_move_times)), 3),
            "cache_snapshot": dict(cache_stats),
        }

        log.info(
            "  RESULT iters=%d  %s=%dW  %s=%dW  draws=%d  "
            "avg_move=%.2fs  total=%.1fs",
            iters, mode_a, wins_a, mode_b, wins_b, draws,
            results[iters]["avg_move_time_s"], elapsed,
        )
        log.info(
            "  Cache  direct=%dH/%dM  move=%dH/%dM",
            cache_stats["direct_hits"],  cache_stats["direct_misses"],
            cache_stats["move_hits"],    cache_stats["move_misses"],
        )

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def print_and_log_summary(results_bd: dict, results_bh: dict):
    log.info("")
    log.info("=" * 80)
    log.info("  FINAL SUMMARY")
    log.info("  %-6s  %-20s  %-20s  %-12s  %-12s",
             "Iters", "baseline vs direct", "baseline vs hybrid",
             "avg_move(bd)", "avg_move(bh)")
    log.info("-" * 80)
    for iters in sorted(results_bd.keys()):
        bd = results_bd[iters]
        bh = results_bh.get(iters, {})
        bd_str = f"{bd['wins_a']}/{bd['wins_b']}/{bd['draws']}"
        bh_str = f"{bh['wins_a']}/{bh['wins_b']}/{bh['draws']}" if bh else "N/A"
        bd_t   = f"{bd['avg_move_time_s']:.2f}s"
        bh_t   = f"{bh.get('avg_move_time_s', 0):.2f}s" if bh else "N/A"
        log.info("  %-6d  %-20s  %-20s  %-12s  %-12s",
                 iters, bd_str, bh_str, bd_t, bh_t)
    log.info("=" * 80)

    summary = {
        "run_id": run_id,
        "config": vars(args),
        "baseline_vs_direct": results_bd,
        "baseline_vs_hybrid": results_bh,
        "final_cache": dict(cache_stats),
    }
    with open(log_json, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("JSON summary → %s", log_json)
    log.info("CSV games    → %s", log_csv_games)
    log.info("CSV moves    → %s", log_csv)
    log.info("Full log     → %s", log_txt)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    res_bd = run_experiment("baseline", "direct",
                            args.iters, args.games,
                            args.hybrid_k, args.llm_iters_scale)

    res_bh = run_experiment("baseline", "hybrid",
                            args.iters, args.games,
                            args.hybrid_k, args.llm_iters_scale)

    print_and_log_summary(res_bd, res_bh)
    csv_file.close()
    csv_games_file.close()

    from connect4.llm_eval import save_cache
    save_cache()
    log.info("Cache saved → llm_cache.json")