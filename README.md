# black_numba

**Project still in developement**

Numba-enhanced bitboard chess engine in python

Play black_numba on lichess.org: https://lichess.org/@/black_numba

## Board state: Bitboards

A bitboard is a way to represent a chess board, with a 64-bit unsigned integer representing
occupied squares with 1 and empty squares with 0.

A position object store 12 bitboards for each chess piece (pawn, knight, bishop, rook, queen, king) of each color
for example, the white pawn bitboard at the beginning of a game will be:

* in decimal: 71776119061217280
* in binary : 0b11111111000000000000000000000000000000000000000000000000

and in a clearer form:
'''
8  ·  ·  ·  ·  ·  ·  ·  · 
7  ·  ·  ·  ·  ·  ·  ·  · 
6  ·  ·  ·  ·  ·  ·  ·  · 
5  ·  ·  ·  ·  ·  ·  ·  · 
4  ·  ·  ·  ·  ·  ·  ·  · 
3  ·  ·  ·  ·  ·  ·  ·  · 
2  1  1  1  1  1  1  1  1 
1  ·  ·  ·  ·  ·  ·  ·  · 
   A  B  C  D  E  F  G  H
'''


## Search
### Iterative deepening
* Negamax serch
  * Alpha-Beta pruning
  * Principal variation search (PVS)
  * Late move reduction (LMR)
  * Null-move pruning
  * Aspiration windows
* Quiescence search (only captures)

### Move ordering
  * Principal Variation (PV)
  
  For captures:
  * Most Valuable Victim - Least Valuable Aggressor (MMV LVA)
  
  For quiet moves:
  * Killer heuristic
  * History heuristic

## Evaluation

* Material scores
* Piece-square tables
* Double, isolated and passed pawns
* Open and semi-open files for rooks and kings


## Numba

[Numba](https://numba.pydata.org/numba-doc/dev/user/5minguide.html) is an open-source
JIT compiler that translates a subset of Python and NumPy into fast machine code using
LLVM, via the llvmlite Python package.

Perft speed in nodes/second:
* Python:  _ __7 300 n/s 
* Numba:   1 500 000 n/s 


## Name

Black Mambas are a variety of snake 10 times faster than pythons.


## Credit
[Bitboard CHESS ENGINE in C on youtube](https://youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs)

[Chess Programming Wiki](https://www.chessprogramming.org/Main_Page)

[Negamax article on wikipedia](https://en.wikipedia.org/wiki/Negamax)

[Talk Chess Forum](http://talkchess.com/forum3/index.php)
