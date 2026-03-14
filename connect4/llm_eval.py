import os
import json
import logging
import random
import asyncio
# Suppress before client instantiation
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
from openai import OpenAI, AsyncOpenAI
from .connect4 import get_player_to_play, to_state, valid_move, play

# ── Clients (sync for single calls, async for batch) ─────────────────────────
_api_key  = os.environ.get("DEEPSEEK_API_KEY")
_base_url = "https://api.deepseek.com"
client       = OpenAI(      api_key=_api_key, base_url=_base_url)
async_client = AsyncOpenAI( api_key=_api_key, base_url=_base_url)

# ── Persistent cache ──────────────────────────────────────────────────────────
CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "llm_cache.json")

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"direct": {}, "move": {}}

def save_cache():
    """Call this periodically or at exit to persist the cache."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({"direct": _direct_cache, "move": _move_cache}, f)
    except Exception as e:
        pass

_cache_data  = _load_cache()
_direct_cache: dict = _cache_data.get("direct", {})
_move_cache:   dict = _cache_data.get("move",   {})

# ── Stats ─────────────────────────────────────────────────────────────────────
cache_stats = {"direct_hits": 0, "direct_misses": 0,
               "move_hits":   0, "move_misses":   0}

# ── Helpers ───────────────────────────────────────────────────────────────────

def board_to_text(grid) -> str:
    symbols = {0: '.', 1: 'X', -1: 'O'}
    rows = [' '.join(symbols[v] for v in row) for row in grid]
    header = ' '.join(str(i) for i in range(grid.shape[1]))
    return header + '\n' + '\n'.join(rows)

def _build_eval_prompt(grid) -> str:
    player_label = 'X' if get_player_to_play(grid) == 1 else 'O'
    opponent     = 'O' if player_label == 'X' else 'X'
    return f"""You are a Connect Four expert evaluating a mid-game position.

Board (rows top→bottom, columns 0-6):
{board_to_text(grid)}

Current player to move: {player_label}  |  Opponent: {opponent}

Evaluate carefully:
- Are there any 3-in-a-row threats for either player?
- Who controls the center columns?
- Who has more immediate winning threats?

Give the win probability for {player_label} (0.0=certain loss, 0.5=equal, 1.0=certain win).
Use the full range — do NOT default to 0.75.
Reply with ONLY a number, e.g. 0.62"""

def _build_move_prompt(grid) -> str:
    player_label = 'X' if get_player_to_play(grid) == 1 else 'O'
    legal = valid_move(grid)
    return f"""You are a Connect Four expert.

Board (rows top→bottom, columns 0-6):
{board_to_text(grid)}

Legal columns: {legal}
It is {player_label}'s turn.

Which single column gives {player_label} the best outcome?
Reply with ONLY the column number, e.g. 3"""

def _call_deepseek(prompt: str, max_tokens: int = 16) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

async def _async_call(prompt: str, max_tokens: int = 16) -> str:
    response = await async_client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


# ── Async batch evaluator ─────────────────────────────────────────────────────

async def _batch_eval_async(state_keys: list, prompts: list) -> list:
    """Fire all uncached prompts concurrently, return scores in order."""
    tasks = [_async_call(p, max_tokens=10) for p in prompts]
    raws  = await asyncio.gather(*tasks, return_exceptions=True)
    scores = []
    for raw in raws:
        try:
            s = float(raw) if not isinstance(raw, Exception) else 0.5
            scores.append(max(0.0, min(1.0, s)))
        except Exception:
            scores.append(0.5)
    return scores

def batch_evaluate_direct(grids: list) -> list:
    """
    Evaluate a list of grids in parallel.
    Returns list of float scores, one per grid.
    Cache hits are resolved instantly; misses are batched into one async round.
    """
    results  = [None] * len(grids)
    need_idx = []   # indices that need an API call
    prompts  = []

    for i, grid in enumerate(grids):
        key = to_state(grid)
        if key in _direct_cache:
            cache_stats["direct_hits"] += 1
            results[i] = _direct_cache[key]
        else:
            cache_stats["direct_misses"] += 1
            need_idx.append(i)
            prompts.append(_build_eval_prompt(grid))

    if prompts:
        scores = asyncio.run(_batch_eval_async(
            [to_state(grids[i]) for i in need_idx], prompts
        ))
        for idx, score in zip(need_idx, scores):
            key = to_state(grids[idx])
            _direct_cache[key] = score
            results[idx] = score

    return results


# ── Method 1 — Direct position evaluation (single, sync) ─────────────────────

def llm_evaluate_direct(grid) -> float:
    key = to_state(grid)
    if key in _direct_cache:
        cache_stats["direct_hits"] += 1
        return _direct_cache[key]

    cache_stats["direct_misses"] += 1
    try:
        raw   = _call_deepseek(_build_eval_prompt(grid), max_tokens=10)
        score = max(0.0, min(1.0, float(raw)))
    except Exception:
        score = 0.5

    _direct_cache[key] = score
    return score

def simulate_direct(grid) -> int:
    score  = llm_evaluate_direct(grid)
    player = get_player_to_play(grid)
    return player if score >= 0.5 else -player


# ── Method 2 — Hybrid rollout ─────────────────────────────────────────────────

def llm_pick_move(grid) -> int:
    key = to_state(grid)
    if key in _move_cache:
        cache_stats["move_hits"] += 1
        return _move_cache[key]

    cache_stats["move_misses"] += 1
    legal = valid_move(grid)
    try:
        raw = _call_deepseek(_build_move_prompt(grid), max_tokens=4)
        col = int(raw)
        if col not in legal:
            col = random.choice(legal)
    except Exception:
        col = random.choice(legal)

    _move_cache[key] = col
    return col

def simulate_hybrid(grid, k: int = 3) -> int:
    from .mcts_ia import random_play_improved
    grid = grid.copy()
    for _ in range(k):
        moves = valid_move(grid)
        if not moves:
            return 0
        player = get_player_to_play(grid)
        col    = llm_pick_move(grid)
        grid, winner = play(grid, col)
        if abs(winner) > 0:
            return player
    return random_play_improved(grid)