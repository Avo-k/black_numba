# black_numba

**Project still in developement**

Numba-enhanced bitboard chess engine in python

* Bitboards

* Iterative deepening
  * Negamax serch
    * Alpha-Beta pruning
  * Quiescence search (only captures)

* Move ordering
  * Most Valuable Victim - Least Valuable Aggressor (MMV LVA)
  * Principal Variation (PV)



Black Mambas are a variety of snake 10 times faster than pythons.

current perft speed: 1.2M nodes/second (7300 n/s without numba)
in-game speed :      150k nodes/second


## Credit
[Bitboard CHESS ENGINE in C on youtube](https://youtube.com/playlist?list=PLmN0neTso3Jxh8ZIylk74JpwfiWNI76Cs)

[Chess Programming Wiki](https://www.chessprogramming.org/Main_Page)

[Negamax article on wikipedia](https://en.wikipedia.org/wiki/Negamax)

[Talk Chess Forum](http://talkchess.com/forum3/index.php)
