import numpy as np
import numba as nb
from numba import njit
from bb_operations import print_bb

EMPTY = np.uint64(0)
BIT = np.uint64(1)
UNIVERSE = np.uint64(0xffffffffffffffff)

white, black, both = np.arange(3, dtype=np.uint8)

pawn, knight, bishop, rook, queen, king = range(6)
# piece_names = ("pawn", "knight", "bishop", "rook", "queen", "king")

a8, b8, c8, d8, e8, f8, g8, h8, \
a7, b7, c7, d7, e7, f7, g7, h7, \
a6, b6, c6, d6, e6, f6, g6, h6, \
a5, b5, c5, d5, e5, f5, g5, h5, \
a4, b4, c4, d4, e4, f4, g4, h4, \
a3, b3, c3, d3, e3, f3, g3, h3, \
a2, b2, c2, d2, e2, f2, g2, h2, \
a1, b1, c1, d1, e1, f1, g1, h1, no_sq = range(65)
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

# Rank masks
rank8, rank7, rank6, rank5, rank4, rank3, rank2, rank1 = \
    np.array([0x00000000000000FF << 8 * i for i in range(8)], dtype=np.uint64)

RANKS = np.array((rank8, rank7, rank6, rank5, rank4, rank3, rank2, rank1))

# File masks
fileA, fileB, fileC, fileD, fileE, fileF, fileG, fileH = \
    np.array([0x0101010101010101 << i for i in range(8)], dtype=np.uint64)

FILES = np.array((fileA, fileB, fileC, fileD, fileE, fileF, fileG, fileH))

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

# Hash Constants

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

# Evaluation Constants

# Material values

material_score = (100, 320, 330, 500, 950, 12000)

mg_phase_score = 6000
eg_phase_score = 500

pawn_pst = (
    0, 0, 0, 0, 0, 0, 0, 0,
    -10, -4, 0, -5, -5, 0, -4, -10,
    -10, -4, 0, 8, 5, 0, -4, -10,
    -10, -4, 0, 16, 12, 0, -4, -10,
    -10, -4, 0, 18, 14, 0, -4, -10,
    -10, -4, 0, 17, 13, 0, -4, -10,
    -10, -4, 0, 16, 12, 0, -4, -10,
    0, 0, 0, 0, 0, 0, 0, 0)

knight_pst = (
    -20, 0, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 3, 3, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 3, 3, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20)

bishop_pst = (
    -2, -2, -2, -2, -2, -2, -2, -2,
    -2, 8, 5, 5, 5, 5, 8, -2,
    -2, 3, 3, 5, 5, 3, 3, -2,
    -2, 2, 5, 4, 4, 5, 2, -2,
    -2, 2, 5, 4, 4, 5, 2, -2,
    -2, 3, 3, 5, 5, 3, 3, -2,
    -2, 8, 5, 5, 5, 5, 8, -2,
    -2, -2, -2, -2, -2, -2, -2, -2)

rook_pst = (
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0)

king_pst = (
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 5, 5, 5, 5, 0, 0,
    0, 5, 5, 10, 10, 5, 5, 0,
    0, 5, 10, 20, 20, 10, 5, 0,
    0, 5, 10, 20, 20, 10, 5, 0,
    0, 0, 5, 10, 10, 5, 0, 0,
    0, 5, 5, -5, -5, 0, 5, 0,
    0, 0, 5, 0, -15, 0, 10, 0)

PST = np.array((pawn_pst, knight_pst, bishop_pst, rook_pst, np.zeros(64), king_pst), dtype=np.int8)

mirror_pst = (
    a1, b1, c1, d1, e1, f1, g1, h1,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a8, b8, c8, d8, e8, f8, g8, h8)

"""
    Masks

          Rank mask            File mask           Isolated mask      White Passed pawn mask
        for square a6        for square f2         for square g2          for square c4
    8  . . . . . . . .    8  . . . . . 1 . .    8  . . . . . 1 . 1     8  . 1 1 1 . . . .
    7  . . . . . . . .    7  . . . . . 1 . .    7  . . . . . 1 . 1     7  . 1 1 1 . . . .
    6  1 1 1 1 1 1 1 1    6  . . . . . 1 . .    6  . . . . . 1 . 1     6  . 1 1 1 . . . .
    5  . . . . . . . .    5  . . . . . 1 . .    5  . . . . . 1 . 1     5  . 1 1 1 . . . .
    4  . . . . . . . .    4  . . . . . 1 . .    4  . . . . . 1 . 1     4  . . . . . . . .
    3  . . . . . . . .    3  . . . . . 1 . .    3  . . . . . 1 . 1     3  . . . . . . . .
    2  . . . . . . . .    2  . . . . . 1 . .    2  . . . . . 1 . 1     2  . . . . . . . .
    1  . . . . . . . .    1  . . . . . 1 . .    1  . . . . . 1 . 1     1  . . . . . . . .
       a b c d e f g h       a b c d e f g h       a b c d e f g h        a b c d e f g h
"""

file_masks, rank_masks, isolated_masks, white_passed_masks, black_passed_masks = \
    (np.zeros(64, dtype=np.uint64) for _ in range(5))

pointer = 0
for i_rank, rank in enumerate(RANKS):
    for i_file, file in enumerate(FILES):
        rank_masks[pointer] = rank
        file_masks[pointer] = file

        if i_file == 0:  # A file
            isolated_masks[pointer] = fileB
            white_passed_masks[pointer] = fileA | fileB
            black_passed_masks[pointer] = fileA | fileB

        elif i_file == 7:  # H file
            isolated_masks[pointer] = fileG
            white_passed_masks[pointer] = fileG | fileH
            black_passed_masks[pointer] = fileG | fileH

        else:
            isolated_masks[pointer] = FILES[i_file - 1] | FILES[i_file + 1]
            white_passed_masks[pointer] = FILES[i_file - 1] | file | FILES[i_file + 1]
            black_passed_masks[pointer] = FILES[i_file - 1] | file | FILES[i_file + 1]

        for r in RANKS[:i_rank + 1]:
            black_passed_masks[pointer] &= ~r
        for r in RANKS[i_rank:]:
            white_passed_masks[pointer] &= ~r

        pointer += 1
del pointer

double_pawn_penalty = -20
isolated_pawn_penalty = -10
passed_pawn_bonus = (200, 150, 100, 75, 50, 30, 10, 0)

semi_open_file_bonus = 10
open_file_bonus = 20

king_shield_bonus = 5
