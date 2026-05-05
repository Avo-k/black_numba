# black_numba

Far from my first chess engine [Skormfish](https://github.com/Avo-k/skormfish)
written in a clear pythonic style, black_numba is a Numba-enhanced bitboard
chess engine written with performance in mind.

My goal is in a 1st time to make a strong engine in python and then in a 2nd
to shape its play-style to make it very aggressive and fun-to-play-against for
humans. Even if it has to become weaker in the process, a hustler engine which
sacrifice pieces for tempo and mobility and try to dirty flag you is my final
goal.

In this readme I will document the project and try to make it as easy as
possible to understand for beginners in chess programming.

## Lichess

**Play black_numba on lichess:** https://lichess.org/@/black_numba

## Quickstart

The project is managed with [uv](https://docs.astral.sh/uv/). Python 3.10–3.13
is supported (Numba's current ceiling); the project pins 3.12 by default.

```bash
# install dependencies into a local .venv
uv sync

# play in your terminal as a UCI engine
uv run black-numba-uci

# run perft to verify the move generator (≈ 7–8 Mn/s on a modern laptop)
uv run python -m black_numba.perft
```

Plug `uv run black-numba-uci` into any UCI-compatible GUI (Cute Chess, Arena,
Banksia, lichess-bot, …) as the engine command and you're good to go.

### Lichess bot mode

`lichess.py` connects black_numba to a [Lichess bot account](https://lichess.org/api#tag/Bot)
through [berserk](https://github.com/lichess-org/berserk). It accepts standard /
from-position challenges in bullet, blitz, rapid and classical, plays them, and
loops.

```bash
export LICHESS_API_TOKEN=lip_xxx          # bot token, scope bot:play
uv run black-numba-lichess
```

A token can also be read from `api_token.txt` in the working directory if the
env var is not set. An optional polyglot opening book is loaded from
`book/Daring.bin` (path overridable with `BLACK_NUMBA_BOOK`); the bot
gracefully skips opening theory when the file is missing.

#### About the opening book

`Daring.bin` is one of [Jeroen Noomen](https://www.chessprogramming.org/Jeroen_Noomen)'s
hand-tuned polyglot books. Noomen is a Dutch opening theorist who has spent
two decades curating books for the top chess engines (Rybka, Komodo,
Stockfish, …) — they're a staple of the TCEC scene. Rather than scraping
master games statistically, he picks lines move by move, biasing for sound
play with active piece counterplay.

The "Daring" set leans into the sharpest mainstream replies: gambits where
they're sound, sicilians, king's indians, dragons. It's a great fit for the
hustler style this engine is reaching for — provocative openings that put
pressure on the human, even at the cost of some quiet equality. A sister
file, `Variety.bin`, is also available in the `book/` directory if you'd
rather avoid playing the exact same lines every game (`BLACK_NUMBA_BOOK=book/Variety.bin`).

See [Deployment](#deployment) below for running the bot on a home server.

## Deployment

The repo ships a [`Dockerfile`](Dockerfile) (multi-stage, uv-based) and a
[`compose.yaml`](compose.yaml) so the bot can run unattended on any Docker
host — typically an LXC container or a small VM on Proxmox.

```bash
# from the repo on the server
export LICHESS_API_TOKEN=lip_xxx
docker compose up -d --build
docker compose logs -f black-numba   # follow the engine output
```

The container restarts automatically (`restart: unless-stopped`). On the first
move after a (re)start, it spends ~30 s JIT-compiling — challenges that arrive
during that window queue up on the Lichess side and play normally as soon as
warmup finishes.

Two polyglot opening books (`Daring.bin`, `Variety.bin`) are bundled with the
image at `/app/book/`. The bot defaults to `Daring.bin`; set
`BLACK_NUMBA_BOOK=book/Variety.bin` (or mount your own book and point the env
var at it) to switch.

## Board state using bitboards

A position object is composed of:
* 12 pieces bitboards
* 3 occupancy bitboards
* Side to move
* En passant square (optional)
* Castling rights
* Hash key

### Bitboards

A [bitboard](https://www.chessprogramming.org/Bitboard_Board-Definition)
is a way to represent a chess board, with a 64-bit unsigned
integer representing occupied squares with 1 and empty squares with 0.

A position object stores 12 piece bitboards, 1 for each chess piece
(pawn, knight, bishop, rook, queen, king) of each color,
and 3 occupancy bitboards (white, black, both) for move generation purpose.

e.g., the white pawn bitboard at the beginning of a game will be:

* in decimal: `71776119061217280`
* in binary: `0b11111111000000000000000000000000000000000000000000000000`

and in a clearer form with coordinates and zeros as dots for readability:
```
8  ·  ·  ·  ·  ·  ·  ·  ·
7  ·  ·  ·  ·  ·  ·  ·  ·
6  ·  ·  ·  ·  ·  ·  ·  ·
5  ·  ·  ·  ·  ·  ·  ·  ·
4  ·  ·  ·  ·  ·  ·  ·  ·
3  ·  ·  ·  ·  ·  ·  ·  ·
2  1  1  1  1  1  1  1  1
1  ·  ·  ·  ·  ·  ·  ·  ·
   A  B  C  D  E  F  G  H
   ```

and the same goes for the 12 others types of pieces:

![alt text](https://github.com/Avo-k/black_numba/blob/master/logo/pieces_bitboard.gif?raw=true)

credit: CPW

### Side to move

The side to move is 0 (white) or 1 (black).
This method allows to easily switch to opponent side using the bitwise XOR operator:
`side ^= 1`

### En passant square

En passant square is stored as an integer. Squares are refered as numbers from 0 to 63
(no_square is 64) using the
[big-endian rank-file mapping](https://www.chessprogramming.org/Big-endian),
which means we see the board from top left to bottom right, illustration below:

   ```
0  1  2  3  4  5  6  7          A8 B8 C8 D8 E8 F8 G8 H8
8  9  10 11 12 13 14 15         A7 B7 C7 D7 E7 F7 G7 H7
16 17 18 19 20 21 22 23         A6 B6 C6 D6 E6 F6 G6 H6
24 25 26 27 28 29 30 31    =    A5 B5 C5 D5 E5 F5 G5 H5
32 33 34 35 36 37 38 39         A4 B4 C4 D4 E4 F4 G4 H4
40 41 42 43 44 45 46 47         A3 B3 C3 D3 E3 F3 G3 H3
48 49 50 51 52 53 54 55         A2 B2 C2 D2 E2 F2 G2 H2
56 57 58 59 60 61 62 63         A1 B1 C1 D1 E1 F1 G1 H1
   ```

### Casteling rights

4 bits are used to represent casteling rights and are printed as in a fen representation `KQkq`

   ```
    bin  dec
   0001    1  white king can castle to the king side
   0010    2  white king can castle to the queen side
   0100    4  black king can castle to the king side
   1000    8  black king can castle to the queen side
   ```

### Hash key (zobrist hashing)

The hash key is a 64-bit integer which will be used as a unique key to store the position
in the hash table alias [transposition table](https://www.chessprogramming.org/Transposition_Table).

First we initialize some random integer for all pieces on all squares, every square
(for en-passant moves), for each castle-right combination, and for the side to move.

Then we simply XOR the corresponding key with each change with the current hash key of the position,
for example when we switch sides:
   `position.hash_key ^= side_key`

## Move generation

Moves are encoded into a single 64-bit integer that packs source square, target
square, piece, promoted piece, capture / double-push / en-passant / castling
flags. This avoids allocating a `Move` object per node and keeps move lists
trivially copyable inside Numba-jitted code.

Sliding-piece attacks (bishop, rook, queen) are computed with
[magic bitboards](https://www.chessprogramming.org/Magic_Bitboards): the
magic numbers and per-square attack tables in `attack_tables.py` are
pre-computed once at import time and reused for the lifetime of the process.

Move generation is verified against [python-chess](https://python-chess.readthedocs.io/)
through `perft.py`, which runs the six standard
[perft positions](https://www.chessprogramming.org/Perft_Results) and asserts
the node counts match the reference values up to depth 4.

## Search

In order to search the best move in a given chess position, traditionnal chess
engines like Stockfish or black_numba have to explore the tree representing
all legal moves at the highest depth possible, this is called [depth first search](https://www.chessprogramming.org/Depth-First).

### Chess' tree span

When playing chess, the colossal number of moves and games possible forces us to find lots of tricks
to reduce, the number of nodes (game state) we will explore. Otherwise we would be stuck at very low depth.
A few numbers to give you a better idea of why we have to use tricks to "prune off" branches of this tree to
reduce the number of nodes (board position) to explore:

When starting a game, white player has to choose between 20 legal moves. Moving each pawn
1 or 2 squares or one of their knights. Black has the same range of choice and it's white
to play again. We say we are turn 1, or 2 plies into the game, or depth 2. And already,
there could be 400 unique possible positions on the board, of course the number of possible
unique leafs grows faster each time and will be multiplied at each ply by the number of
legal move available.

`10**120` is [The Shannon number](https://en.wikipedia.org/wiki/Shannon_number),
a well known number representing the game-tree complexity of chess, what most
people don't know is that it is the lower bound of the estimation, made in 1950
by the American mathematician Claude Shannon. One way to begin to grasp the absurdity
of this number is to compare it to the number of ATOMS in our observable universe,
which is "only" `10**80`.

### Negamax search

The negamax search is a simple way to explore the tree representing all the possible continuation
of a chess game. In simple words we will explore all the legal moves available to a given depth
using a Depth First Search algorithm and evaluating the position thanks to the evaluation function
when reaching a leaf.
We try guessing which path is the most likely by stating both player:
* wants to maximize their score
* can see at the same depth

[Step-by-step illustration of the negamax algorithm](https://en.wikipedia.org/wiki/Negamax#/media/File:Plain_Negamax.gif)

#### Alpha-Beta pruning

[Alpha-beta pruning](https://www.chessprogramming.org/Alpha-Beta) is the
backbone optimisation: we keep a lower bound `alpha` (best score we can already
guarantee) and an upper bound `beta` (best score the opponent will allow). As
soon as we find a move that lets the opponent score `>= beta`, we can stop
searching the rest of the children — the opponent would never let us reach
this position. Empirically this turns the `O(b^d)` tree into roughly `O(b^(d/2))`
when moves are ordered well, which is why ordering matters so much.

### Iterative deepening

Rather than committing to a single target depth, we search at depth 1, then 2,
then 3, … reusing each iteration's principal variation to seed move ordering at
the next depth. The benefits are:

* we always have a "best move so far" available when the time runs out,
* deeper iterations are dramatically cheaper because the previous PV is searched
  first and triggers cutoffs early,
* time control falls out for free: we just stop launching the next iteration
  once we've used our budget.

The negamax driver is augmented with:

* Alpha-Beta pruning
* Principal Variation Search (PVS) — assume the first move is best, search the
  rest with a null window, only re-search if a move surprises us.
* Late Move Reduction (LMR) — search "late" quiet moves at reduced depth; if
  one ends up looking promising, re-search at full depth.
* Null-move pruning — let the opponent move twice; if we're still doing fine,
  the position is so good we can prune.
* Aspiration windows — open the next iteration with a narrow window around the
  previous score and widen on a fail-high / fail-low.
* Quiescence search — at depth 0 we keep extending capture lines only, so we
  don't return a static evaluation in the middle of a tactical exchange.

### Move ordering

A good move ordering can turn a hopeless search into a fast one. We score
candidate moves before sorting:

* **Hash table** — the previous best move stored for this position via Zobrist hash.
* **Principal Variation (PV)** — the PV move from the previous iteration is tried first.
* Captures, ranked by [Most Valuable Victim − Least Valuable Aggressor (MVV/LVA)](https://www.chessprogramming.org/MVV-LVA).
* Quiet moves:
  * **Killer heuristic** — moves that recently caused a beta cutoff at the same ply.
  * **History heuristic** — `[side][piece][to_square]` table incremented every time a quiet move improves alpha.

### Transposition table

A 4M-entry table indexed by `hash_key % MAX_HASH_SIZE` stores
`(key, depth, flag, score)` per position. Entries record whether the stored
score is exact, an alpha bound, or a beta bound, so repeated positions reached
through different move orders can either return immediately or sharpen the
window without re-searching.

## Evaluation

Evaluation is **tapered**: each term is computed for both the middle-game and the
end-game and blended according to the remaining material (game phase). This
removes the discontinuity you'd otherwise get when a single piece trade flips
the engine from one set of weights to the other.

Implemented terms:

* Material scores (separate MG / EG values)
* [Piece-square tables](https://www.chessprogramming.org/Piece-Square_Tables) (`pst.py`)
* Pawn structure: doubled, isolated and passed pawns
* Open and semi-open files for rooks and kings
* Pawn shield in front of the king
* Piece mobility (count of pseudo-legal squares each minor / major piece reaches)
* Bishop pair bonus
* Tempo bonus
* Piece / king tropism (Manhattan distance weighted by piece value)
* "Anti-mobility" trick for the king (mobility is bad for the king in the middlegame)
* Pawns-on-bishop-colour penalty
* **Lazy eval** — if the running score is already far from the search window
  before computing the expensive terms, return early.

### Endgame tablebases

When fewer than 8 pieces remain on the board, the Lichess bot mode queries the
[Lichess Syzygy tablebase API](http://tablebase.lichess.ovh/) and plays the
DTM-optimal move directly instead of searching.

## Numba

[Numba](https://numba.pydata.org/numba-doc/dev/user/5minguide.html) is an open-source
JIT compiler that translates a subset of Python and NumPy into fast machine code using
LLVM, via the llvmlite Python package.

The hot path (move generation, make-move, attack tables, search, evaluation)
is `@njit`-compiled. The first call pays a one-shot compilation cost (~10–30 s
cold, much less when the on-disk cache from `cache=True` is warm), every call
afterwards runs at compiled speed.

Perft speed in nodes / second on the standard test positions:

| implementation | speed |
| --- | --- |
| pure Python | ~7 300 nps |
| Numba (legacy README) | ~1.5 Mn/s |
| Numba (current, modern CPU) | **~7–8 Mn/s** |

## Name

Black mambas are a variety of snake 10 times faster than pythons.

## Credits
[Bitboard CHESS ENGINE in C on youtube](https://youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs)

[Chess Programming Wiki](https://www.chessprogramming.org/Main_Page)

[Snakefish](https://github.com/cglouch/snakefish/)

[Negamax article on wikipedia](https://en.wikipedia.org/wiki/Negamax)

[Talk Chess Forum](http://talkchess.com/forum3/index.php)
