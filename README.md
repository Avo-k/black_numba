# black_numba

**Project still in development**

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

A bitboard is a way to represent a chess board, with a 64-bit unsigned 
integer representing occupied squares with 1 and empty squares with 0.

A position object stores 12 piece bitboards, 1 for each chess piece 
(pawn, knight, bishop, rook, queen, king) of each color,
and 3 occupancy bitboards (white, black, both) for move generation purpose.

e.g., the white pawn bitboard at the beginning of a game will be:

* in decimal: `71776119061217280`
* in binary: `0b11111111000000000000000000000000000000000000000000000000`

and in a clearer form with coordinates and zeros as dots readability:
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

### Side to move

The side to move is 0 (white) or 1 (black). 
This method allows to use to easily switch side using the bitwise XOR operator:

e.g. side ^= 1

### En passant square

En passant square is stored as an integer. Squares are refered as numbers from 0 to 63
(no_square is 64) using the big-endian rank-file mapping, which means we see the board
from top left to bottom right, illustration below:

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

illustration by Code Monkey King:
   ```
    bin  dec
   0001    1  white king can castle to the king side
   0010    2  white king can castle to the queen side
   0100    4  black king can castle to the king side
   1000    8  black king can castle to the queen side
   
      examples
   1111       both sides an castle both directions
   1001       black king => queen side
              white king => king side
   ```

## Search
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
