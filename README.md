# black_numba

**Project still in developement**

Numba-enhanced bitboard chess engine in python

## Board state
* Bitboards

## Search

* Iterative deepening
* Negamax serch
  * Alpha-Beta pruning
* Quiescence search (only captures)

* Move ordering
  * Most Valuable Victim - Least Valuable Aggressor (MMV LVA)
  * Principal Variation (PV)


## Numba

[Numba](https://numba.pydata.org/numba-doc/dev/user/5minguide.html) is an open-source JIT compiler that translates a subset of Python and NumPy into fast machine code using LLVM, via the llvmlite Python package.
Thanks to Numba, my bitboard code runs 165 times faster in perft test.

Perft speed in nodes/second:
* Python: _____7 300 n/s 
* Numba:  1 200 000 n/s 


## Name

Black Mambas are a variety of snake 10 times faster than pythons.


## Credit
[Bitboard CHESS ENGINE in C on youtube](https://youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs)

[Chess Programming Wiki](https://www.chessprogramming.org/Main_Page)

[Negamax article on wikipedia](https://en.wikipedia.org/wiki/Negamax)

[Talk Chess Forum](http://talkchess.com/forum3/index.php)
