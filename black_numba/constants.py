import numpy as np
import numba as nb
from numba import njit

EMPTY = np.uint64(0)
BIT = np.uint64(1)
UNIVERSE = np.uint64(0xFFFFFFFFFFFFFFFF)

white, black, both = np.arange(3, dtype=np.uint8)

pawn, knight, bishop, rook, queen, king = range(6)

(
    a8,    b8,    c8,    d8,    e8,    f8,    g8,    h8,
    a7,    b7,    c7,    d7,    e7,    f7,    g7,    h7,
    a6,    b6,    c6,    d6,    e6,    f6,    g6,    h6,
    a5,    b5,    c5,    d5,    e5,    f5,    g5,    h5,
    a4,    b4,    c4,    d4,    e4,    f4,    g4,    h4,
    a3,    b3,    c3,    d3,    e3,    f3,    g3,    h3,
    a2,    b2,    c2,    d2,    e2,    f2,    g2,    h2,
    a1,    b1,    c1,    d1,    e1,    f1,    g1,    h1,
    no_sq,
) = np.arange(65, dtype=np.uint8)
squares = range(64)

black_squares = np.array(
    sorted(
        [s for s in range(1, 64, 2) if not (s // 8) % 2]
        + [s for s in range(0, 64, 2) if (s // 8) % 2]
    ),
    dtype=np.uint8,
)
white_squares = np.array(
    sorted(
        [s for s in range(0, 64, 2) if not (s // 8) % 2]
        + [s for s in range(1, 64, 2) if (s // 8) % 2]
    ),
    dtype=np.uint8,
)

square_to_coordinates = (
    "a8",    "b8",    "c8",    "d8",    "e8",    "f8",    "g8",    "h8",
    "a7",    "b7",    "c7",    "d7",    "e7",    "f7",    "g7",    "h7",
    "a6",    "b6",    "c6",    "d6",    "e6",    "f6",    "g6",    "h6",
    "a5",    "b5",    "c5",    "d5",    "e5",    "f5",    "g5",    "h5",
    "a4",    "b4",    "c4",    "d4",    "e4",    "f4",    "g4",    "h4",
    "a3",    "b3",    "c3",    "d3",    "e3",    "f3",    "g3",    "h3",
    "a2",    "b2",    "c2",    "d2",    "e2",    "f2",    "g2",    "h2",
    "a1",    "b1",    "c1",    "d1",    "e1",    "f1",    "g1",    "h1",
    "-",
)

# Rank masks
rank8, rank7, rank6, rank5, rank4, rank3, rank2, rank1 = np.array(
    [0x00000000000000FF << 8 * i for i in range(8)], dtype=np.uint64
)

RANKS = np.array((rank8, rank7, rank6, rank5, rank4, rank3, rank2, rank1))

# File masks
fileA, fileB, fileC, fileD, fileE, fileF, fileG, fileH = np.array(
    [0x0101010101010101 << i for i in range(8)], dtype=np.uint64
)

FILES = np.array((fileA, fileB, fileC, fileD, fileE, fileF, fileG, fileH))
piece_to_letter = (("P", "N", "B", "R", "Q", "K"), ("p", "n", "b", "r", "q", "k"))
piece_to_ascii = (("♟", "♞", "♝", "♜", "♛", "♚"), ("♙", "♘", "♗", "♖", "♕", "♔"))

wk, wq, bk, bq = (2 ** i for i in range(4))
castling_rights = np.array([15 for _ in range(64)], dtype=np.uint8)
castling_rights[:8] = (7, 15, 15, 15, 3, 15, 15, 11)
castling_rights[-8:] = (13, 15, 15, 15, 12, 15, 15, 14)

empty_board = "8/8/8/8/8/8/8/8 w - - "
start_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
tricky_position = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
killer_position = "rnbqkb1r/pp1p1pPp/8/2p1pP2/1P1P4/3P3P/P1P1P3/RNBQKBNR w KQkq e6 0 1"
cmk_position = "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9"
repetitions_position = "2r3k1/R7/8/1R6/8/8/P4KPP/8 w - - 0 40"
mate_in_3 = "1k6/6R1/3K4/8/8/8/8/8 w - - 18 10"
mate_in_2 = "k7/6R1/2K5/8/8/8/8/8 w - - 16 9"
mate_in_4 = "2k5/5R2/3K4/8/8/8/8/8 w - - 12 7"
bnk_mate = "1k6/8/8/8/8/8/8/1K2BN2 w - - 0 1"

FENS = (
    start_position,
    tricky_position,
    killer_position,
    cmk_position,
    repetitions_position,
    mate_in_2,
    mate_in_4,
)

# Mating bounds for mating scores
BOUND_INF, UPPER_MATE, LOWER_MATE = 50000, 49000, 48000

# Capture ordering
# most valuable victim & less valuable attacker
# MVV LVA [attacker][victim]
mvv_lva = np.array((
        (105, 205, 305, 405, 505, 605),
        (104, 204, 304, 404, 504, 604),
        (103, 203, 303, 403, 503, 603),
        (102, 202, 302, 402, 502, 602),
        (101, 201, 301, 401, 501, 601),
        (100, 200, 300, 400, 500, 600),
    ))

# PV
MAX_PLY = 64

# LMR
full_depth_moves = 4
reduction_limit = 3

# Time
time_precision = 2047

# Hash Constants
# init random hash keys
piece_keys = np.random.randint(2 ** 64 - 1, size=(2, 6, 64), dtype=np.uint64)
en_passant_keys = np.random.randint(2 ** 64 - 1, size=64, dtype=np.uint64)
castle_keys = np.random.randint(2 ** 64 - 1, size=16, dtype=np.uint64)
side_key = np.random.randint(2 ** 64 - 1, dtype=np.uint64)

MAX_HASH_SIZE = 0x400000

hash_flag_exact, hash_flag_alpha, hash_flag_beta = range(3)
no_hash_entry = 100000

hash_numpy_type = np.dtype(
    [("key", np.uint64), ("depth", np.uint8), ("flag", np.uint8), ("score", np.int64)]
)
hash_numba_type = nb.from_dtype(hash_numpy_type)

# Evaluation Constants

# Material values           Middle-game                         Endgame
#                    P   N    B    R    Q     K       P   N    B    R    Q     K
material_scores = ((70, 325, 325, 500, 975, 12000), (90, 315, 315, 500, 975, 12000))

#               P  N  B  R  Q  K
phase_scores = (0, 1, 1, 2, 4, 0)
TOTAL_PHASE = 24

mg_phase_score = 6000
eg_phase_score = 1000

opening, end_game, middle_game = np.arange(3, dtype=np.uint8)

mirror_pst = (
    a1,    b1,    c1,    d1,    e1,    f1,    g1,    h1,
    a2,    b2,    c2,    d2,    e2,    f2,    g2,    h2,
    a3,    b3,    c3,    d3,    e3,    f3,    g3,    h3,
    a4,    b4,    c4,    d4,    e4,    f4,    g4,    h4,
    a5,    b5,    c5,    d5,    e5,    f5,    g5,    h5,
    a6,    b6,    c6,    d6,    e6,    f6,    g6,    h6,
    a7,    b7,    c7,    d7,    e7,    f7,    g7,    h7,
    a8,    b8,    c8,    d8,    e8,    f8,    g8,    h8,
)

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

file_masks, rank_masks, isolated_masks, white_passed_masks, black_passed_masks = (
    np.zeros(64, dtype=np.uint64) for _ in range(5)
)


def init_masks():
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

            for r in RANKS[: i_rank + 1]:
                black_passed_masks[pointer] &= ~r
            for r in RANKS[i_rank:]:
                white_passed_masks[pointer] &= ~r

            pointer += 1


init_masks()

double_pawn_penalty = -20
isolated_pawn_penalty = -10
passed_pawn_bonus = (200, 150, 100, 75, 50, 30, 10, 0)

semi_open_file_bonus = 10
open_file_bonus = 20

king_shield_bonus = 5

bishop_pair_mg = 50
bishop_pair_eg = 70

knight_tropism_mg = 3
bishop_tropism_mg = 2
rook_tropism_mg = 2
queen_tropism_mg = 2

knight_tropism_eg = 1
bishop_tropism_eg = 1
rook_tropism_eg = 1
queen_tropism_eg = 4

pawns_on_bishop_colour_opening = (9, 6, 3, 0, -3, -6, -9, -12, -15)
pawns_on_bishop_colour_endgame = (12, 8, 4, 0, -4, -8, -12, -16, -20)


@njit
def manhattan_distance(sq1, sq2):
    F1, F2 = sq1 & 7, sq2 & 7
    R1, R2 = sq1 >> 3, sq2 >> 3
    return abs(R2 - R1) + abs(F2 - F1)


arr_manhattan = np.array(
    [manhattan_distance(sq1, sq2) for sq1 in range(64) for sq2 in range(64)],
    dtype=np.uint8,
)
arr_manhattan.shape = (64, 64)

stopped = False
