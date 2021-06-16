import numpy as np
import numba as nb
from numba import njit

EMPTY = np.uint64(0)
BIT = np.uint64(1)
UNIVERSE = np.uint64(0xffffffffffffffff)

white, black, both = range(3)
pc_white, pc_black = False, True

pawn, knight, bishop, rook, queen, king = range(6)
pc_pawn, pc_knight, pc_bishop, pc_rook, pc_queen, pc_king = np.arange(1, 7)
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

mirror_pst = (
    a1, b1, c1, d1, e1, f1, g1, h1,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a8, b8, c8, d8, e8, f8, g8, h8)

RANKS = np.array(
    [0x00000000000000FF << 8 * i for i in range(8)],
    dtype=np.uint64)

FILES = np.array(
    [0x0101010101010101 << i for i in range(8)],
    dtype=np.uint64)

# File
fileA, fileB, fileC, fileD, fileE, fileF, fileG, fileH = np.arange(8, dtype=np.uint8)

piece_to_letter = [{0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K'},
                   {0: 'p', 1: 'n', 2: 'b', 3: 'r', 4: 'q', 5: 'k'}]

letter_to_piece = [{'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5},
                   {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}]

piece_to_ascii = [{0: '♟', 1: '♞', 2: '♝', 3: '♜', 4: '♛', 5: '♚'},
                  {0: '♙', 1: '♘', 2: '♗', 3: '♖', 4: '♕', 5: '♔'}]

promo_piece_to_str_ = {4: 'q', 3: 'r', 2: 'b', 1: 'n'}
promo_piece_to_str = ['p', 'n', 'b', 'r', 'q', 'k']
# promo_str_to_piece = {v: k for k, v in promo_piece_to_str.items()}


wk, wq, bk, bq = (np.uint8(2 ** i) for i in range(4))

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

# Capture ordering
# most valuable victim & less valuable attacker
# MVV LVA [attacker][victim]
mvv_lva = np.array(((105, 205, 305, 405, 505, 605),
                    (104, 204, 304, 404, 504, 604),
                    (103, 203, 303, 403, 503, 603),
                    (102, 202, 302, 402, 502, 602),
                    (101, 201, 301, 401, 501, 601),
                    (100, 200, 300, 400, 500, 600)), dtype=np.uint64)

MAX_PLY = 64

full_depth_moves = 4
reduction_limit = 3

# np.random.seed(23)

# init random keys
pieces_keys = np.random.randint(2 ** 64 - 1, size=(2, 6, 64), dtype=np.uint64)
en_passant_keys = np.random.randint(2 ** 64 - 1, size=64, dtype=np.uint64)
castle_keys = np.random.randint(2 ** 64 - 1, size=16, dtype=np.uint64)
side_key = np.random.randint(2 ** 64 - 1, dtype=np.uint64)

