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


### Iterative deepening
* Negamax search
  * Alpha-Beta pruning
  * Principal variation search (PVS)
  * Late move reduction (LMR)
  * Null-move pruning
  * Aspiration windows
  * TODO: lazy search
* Quiescence search (only captures)

### Move ordering
  * Principal Variation (PV)
  
  Captures:
  * Most Valuable Victim - Least Valuable Aggressor (MMV LVA)
  
  Quiet moves:
  * Killer heuristic
  * History heuristic

  All moves:
  * Hash table

## Evaluation

* Material scores
* Piece-square tables
* Double, isolated and passed pawns
* Open and semi-open files for rooks and kings
* Shield-pawn for king
* Pieces mobility
* Bishop pair
* Tempo
* Pieces/ king tropism (Manhattan distance weighted by piece value)
* "anti-mobility" trick for king
  
* Tapered eval to remove evaluation discontinuity
* lazy eval


## Numba

[Numba](https://numba.pydata.org/numba-doc/dev/user/5minguide.html) is an open-source
JIT compiler that translates a subset of Python and NumPy into fast machine code using
LLVM, via the llvmlite Python package.

Perft speed in nodes/second:
* Python: 7 300
* Numba:  1 500 000


## Name

Black mambas are a variety of snake 10 times faster than pythons.


## Credits
[Bitboard CHESS ENGINE in C on youtube](https://youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs)

[Chess Programming Wiki](https://www.chessprogramming.org/Main_Page)

[Snakefish](https://github.com/cglouch/snakefish/)

[Negamax article on wikipedia](https://en.wikipedia.org/wiki/Negamax)

[Talk Chess Forum](http://talkchess.com/forum3/index.php)
