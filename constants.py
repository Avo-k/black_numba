import numpy as np
import numba as nb
from collections import namedtuple

EMPTY = np.uint64(0)
BIT = np.uint64(1)
UNIVERSE = np.uint64(0xffffffffffffffff)

white, black, both = range(3)

pawn, knight, bishop, rook, queen, king = range(6)
piece_names = ["pawn", "knight", "bishop", "rook", "queen", "king"]

a8, b8, c8, d8, e8, f8, g8, h8, a7, b7, c7, d7, e7, f7, g7, h7, \
a6, b6, c6, d6, e6, f6, g6, h6, a5, b5, c5, d5, e5, f5, g5, h5, \
a4, b4, c4, d4, e4, f4, g4, h4, a3, b3, c3, d3, e3, f3, g3, h3, \
a2, b2, c2, d2, e2, f2, g2, h2, a1, b1, c1, d1, e1, f1, g1, h1, no_sq = np.arange(65, dtype=np.uint8)
squares = np.arange(64, dtype=np.uint8)

square_to_coordinates = [
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "-"]

RANKS = np.array(
    [0x00000000000000FF << 8 * i for i in range(8)],
    dtype=np.uint64)

FILES = np.array(
    [0x0101010101010101 << i for i in range(8)],
    dtype=np.uint64)

fifi = namedtuple("File", "A B C D E F G H")
File = fifi._make(range(8))

piece_to_letter = [{0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K'},
                   {0: 'p', 1: 'n', 2: 'b', 3: 'r', 4: 'q', 5: 'k'}]

letter_to_piece = [{'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5},
                   {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}]

piece_to_ascii = [{0: '♟', 1: '♞', 2: '♝', 3: '♜', 4: '♛', 5: '♚'},
                  {0: '♙', 1: '♘', 2: '♗', 3: '♖', 4: '♕', 5: '♔'}]

wk, wq, bk, bq = (np.uint8(2 ** i) for i in range(4))
castle_rook_move = {g1: (h1, f1), c1: (a1, d1), g8: (h8, f8), c8: (a8, d8)}
castling_rights = np.array(
    (7, 15, 15, 15, 3, 15, 15, 11,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     15, 15, 15, 15, 15, 15, 15, 15,
     13, 15, 15, 15, 12, 15, 15, 14),
    dtype=np.uint8)


empty_board = "8/8/8/8/8/8/8/8 w - - "
start_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 "
tricky_position = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1 "
killer_position = "rnbqkb1r/pp1p1pPp/8/2p1pP2/1P1P4/3P3P/P1P1P3/RNBQKBNR w KQkq e6 0 1"
cmk_position = "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9 "
