Connect 4 AI
============

* This little project implements a MCTS (Monte-Carlo tree search) for connect 4
* Just launch the **main.py** and you will be able to play against the AI.
* To play you have to input the column number (between 0 and 6).

============
# Connect Four — MCTS vs LLM-Guided MCTS

A Connect Four AI comparing standard Monte Carlo Tree Search against two LLM-guided variants inspired by the [RAP framework](https://aclanthology.org/2023.emnlp-main.507/) (Hao et al., 2023). Uses the DeepSeek API as the LLM evaluator.

Built on top of [floriangardin/connect4-mcts](https://github.com/floriangardin/connect4-mcts).

---

## Project Structure

```
connect4-mcts/
├── connect4/
│   ├── connect4.py       # Game logic — board, play(), win detection
│   ├── mcts.py           # MCTS node — UCT formula, child selection
│   ├── mcts_ia.py        # Baseline MCTS trainer + random_play_improved rollout
│   ├── llm_eval.py       # LLM evaluation: direct + hybrid, persistent cache
│   └── llm_mcts_ia.py    # Unified MCTS trainer supporting all three modes
├── main.py               # Human vs AI interactive game
├── experiment.py         # Automated agent vs agent experiment runner
├── llm_cache.json        # Persistent LLM response cache (auto-generated)
├── results/              # Experiment logs and CSVs (auto-generated)
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/JustinYe377/connect4-mcts.git
cd connect4-mcts
pip install numpy openai
export DEEPSEEK_API_KEY=your_key_here
```

---

## Play Against the AI

```bash
# Standard MCTS — no API calls needed
python main.py

# LLM direct position evaluation
python main.py --mode direct

# LLM hybrid rollout
python main.py --mode hybrid

# Enable debug logging (shows DeepSeek requests, responses, input validation)
python main.py --mode direct --debug
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `baseline` | `baseline` / `direct` / `hybrid` |
| `--iters` | `50` | MCTS iterations per move |
| `--hybrid_k` | `3` | LLM-guided plies before random takeover (hybrid only) |
| `--debug` | off | Show all DeepSeek calls and input validation |

At the start of each game you will be asked whether to go first. You play as **X**, the AI plays as **O**.

---

## Run Experiments (Agent vs Agent)

Runs baseline MCTS against each LLM mode automatically, alternating first player across games.

```bash
python experiment.py --games 10 --iters 50 200 --llm_iters_scale 0.25
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--games` | `10` | Games per matchup per iteration count |
| `--iters` | `50 200 500` | Iteration counts to test (space-separated) |
| `--hybrid_k` | `3` | LLM plies per hybrid rollout |
| `--llm_iters_scale` | `0.5` | LLM modes get this fraction of `--iters` |
| `--logdir` | `results/` | Directory for output files |

**Recommended starting point** (avoids excessive API costs):

```bash
python experiment.py --games 10 --iters 50 --llm_iters_scale 0.25
```

---

## Output Files

Each run writes to `results/`:

| File | Contents |
|------|----------|
| `run_TIMESTAMP.log` | Timestamped log of every game and final summary |
| `run_TIMESTAMP_games.csv` | One row per game — outcome, cumulative W/D/L rates, timing |
| `run_TIMESTAMP_moves.csv` | One row per move — win rate of chosen move, avg tree depth, move time |
| `run_TIMESTAMP_summary.json` | Machine-readable summary of all matchups |

The LLM response cache (`llm_cache.json`) persists across runs — previously seen board positions are not re-evaluated.

---

## Three MCTS Modes

**`baseline`** — Standard MCTS using `random_play_improved` for rollouts. Detects immediate wins and forced blocks during simulation. No API calls required.

**`direct`** — Replaces the random rollout with a single DeepSeek API call that estimates the current player's win probability [0.0–1.0]. Evaluated positions are cached.

**`hybrid`** — LLM picks the best move for the first `k` plies of each rollout, then hands off to `random_play_improved`. Balances LLM guidance with fast random play for the remainder of the game.

---

## Debug Mode

`--debug` logs every DeepSeek request and response, plus human input validation:

```
[DEBUG] ── DeepSeek REQUEST ─────────────────────────────
[DEBUG]   Board (rows top→bottom, columns 0-6):
[DEBUG]   0 1 2 3 4 5 6
[DEBUG]   . . . . . . .
[DEBUG]   . . . X O . .
[DEBUG] ── DeepSeek RESPONSE: '0.62'
[DEBUG] direct_eval  parsed=0.620  (VALID)
[DEBUG] pick_move    VALID      → col=3  (legal=[0,1,2,3,4,5,6])
[DEBUG] INVALID input — col=9 not in legal=[0,1,2,3,4,5,6], fell back to random
```

---

## Results (50 iters, llm_scale=0.25, 10 games)

| Matchup | Baseline | LLM | Draws | Baseline move time | LLM move time |
|---------|----------|-----|-------|--------------------|---------------|
| vs Direct | **9W** | 1W | 0 | 0.63s | 12.4s |
| vs Hybrid | **7W** | 3W | 0 | 0.56s | 40.5s |

Baseline outperformed both LLM methods at this iteration count. See `results/` for full logs.

---

## References

- Hao, S., et al. (2023). [Reasoning with language model is planning with world model.](https://aclanthology.org/2023.emnlp-main.507/) *EMNLP 2023.*
- Base implementation: [floriangardin/connect4-mcts](https://github.com/floriangardin/connect4-mcts)
- LLM API: [DeepSeek Platform](https://platform.deepseek.com)