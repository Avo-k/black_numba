import numpy as np
import numba as nb
from numba import njit

EMPTY = np.uint64(0)
BIT = np.uint64(1)
UNIVERSE = np.uint64(0xffffffffffffffff)

white, black, both = np.arange(3, dtype=np.uint8)

pawn, knight, bishop, rook, queen, king = range(6)
# piece_names = ("pawn", "knight", "bishop", "rook", "queen", "king")

a8, b8, c8, d8, e8, f8, g8, h8, a7, b7, c7, d7, e7, f7, g7, h7, \
a6, b6, c6, d6, e6, f6, g6, h6, a5, b5, c5, d5, e5, f5, g5, h5, \
a4, b4, c4, d4, e4, f4, g4, h4, a3, b3, c3, d3, e3, f3, g3, h3, \
a2, b2, c2, d2, e2, f2, g2, h2, a1, b1, c1, d1, e1, f1, g1, h1, no_sq = range(65)
squares = range(64)

square_to_coordinates = (
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "-")

RANKS = np.array(
    [0x00000000000000FF << 8 * i for i in range(8)],
    dtype=np.uint64)

FILES = np.array(
    [0x0101010101010101 << i for i in range(8)],
    dtype=np.uint64)

# File
fileA, fileB, fileC, fileD, fileE, fileF, fileG, fileH = np.arange(8, dtype=np.uint8)

piece_to_letter = (('P', 'N', 'B', 'R', 'Q', 'K'),
                   ('p', 'n', 'b', 'r', 'q', 'k'))

piece_to_ascii = (('♟', '♞', '♝', '♜', '♛', '♚'),
                  ('♙', '♘', '♗', '♖', '♕', '♔'))


wk, wq, bk, bq = (2 ** i for i in range(4))

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
start_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
tricky_position = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
killer_position = "rnbqkb1r/pp1p1pPp/8/2p1pP2/1P1P4/3P3P/P1P1P3/RNBQKBNR w KQkq e6 0 1"
cmk_position = "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9"
repetitions_position = "2r3k1/R7/8/1R6/8/8/P4KPP/8 w - - 0 40"
mate_in_2 = "k7/6R1/2K5/8/8/8/8/8 w - - 16 9"
mate_in_4 = "2k5/5R2/3K4/8/8/8/8/8 w - - 12 7"

# Mating bounds for mating scores
BOUND_INF, UPPER_MATE, LOWER_MATE = 50000, 49000, 48000

# Capture ordering
# most valuable victim & less valuable attacker
# MVV LVA [attacker][victim]
mvv_lva = np.array(((105, 205, 305, 405, 505, 605),
                    (104, 204, 304, 404, 504, 604),
                    (103, 203, 303, 403, 503, 603),
                    (102, 202, 302, 402, 502, 602),
                    (101, 201, 301, 401, 501, 601),
                    (100, 200, 300, 400, 500, 600)), dtype=np.uint8)

# PV
MAX_PLY = 64

# LMR
full_depth_moves = 4
reduction_limit = 3

# init random hash keys
piece_keys = np.random.randint(2 ** 64 - 1, size=(2, 6, 64), dtype=np.uint64)
en_passant_keys = np.random.randint(2 ** 64 - 1, size=64, dtype=np.uint64)
castle_keys = np.random.randint(2 ** 64 - 1, size=16, dtype=np.uint64)
side_key = np.random.randint(2 ** 64 - 1, dtype=np.uint64)

MAX_HASH_SIZE = 0x400000

hash_flag_exact, hash_flag_alpha, hash_flag_beta = range(3)
no_hash_entry = 100000

hash_numpy_type = np.dtype([('key', np.uint64), ('depth', np.uint8), ('flag', np.uint8), ('score', np.int64)])
hash_numba_type = nb.from_dtype(hash_numpy_type)
