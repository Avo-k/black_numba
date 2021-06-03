from constants import *
from bb_operations import *
from numba.experimental import jitclass


position_spec = [
    ('pieces', nb.uint64[:,:]),
    ('occupancy', nb.uint64[:]),
    ('side', nb.b1),
    ('enpas', nb.uint8),
    ('castle', nb.uint8)]


# @jitclass(position_spec)
class Position(object):
    def __init__(self):
        self.pieces = np.zeros((2, 6), dtype=np.uint64)  # bb for each color (2) and each piece type (6)
        self.occupancy = np.zeros(3, dtype=np.uint64)  # Combined bitboards for [white, black, both]
        self.side = 0
        self.enpas = np.uint8(0)
        self.castle = 0


def print_position(pos, print_info=False):
    print("\n")
    for rank in range(8):
        r = " "
        for file in range(8):
            sq = np.uint8(rank * 8 + file)

            # piece is white
            if get_bit(pos.occupancy[white], sq):
                for piece, bb in enumerate(pos.pieces[white]):
                    if get_bit(bb, sq):
                        r += f" {piece_to_letter[white][piece]} "
                        # print(f"{piece_names[piece]} on {square_to_coordinates[sq]}")
                        break

            # piece is black
            elif get_bit(pos.occupancy[black], sq):
                for i, bb in enumerate(pos.pieces[black]):
                    if get_bit(bb, sq):
                        r += f" {piece_to_letter[black][i]} "
                        break

            # empty square
            else:
                r += " · "

        # assert len(r) == 25
        print(8 - rank, r)

    print("\n    A  B  C  D  E  F  G  H\n")

    if print_info:
        print("white" if not pos.side else "black", "to move")
        print("en passant:", pos.enpas if pos.enpas else "-")
        casl = f"{'K' if pos.castle & wk else ''}{'Q' if pos.castle & wq else ''}" \
               f"{'k' if pos.castle & bk else ''}{'q' if pos.castle & bq else ''} "
        print("Castling:",casl if casl else "-", "\n")


def parse_fen(fen: str):
    """return a Position object from a Forsyth-Edwards Notation string"""

    pos = Position()

    board, side, castle, enpas, _hclock, _fclock = fen.split()

    pos.side = 0 if side == "w" else 1
    pos.enpas = None if enpas == "-" else square_to_coordinates.index(enpas)

    if castle != "-":
        for i, c in enumerate("KQkq"):
            if c in castle:
                pos.castle += (2**i)
    else:
        pos.castle = 0

    squares = np.arange(64, dtype=np.uint8)
    sq_i = 0

    for c in board:
        if c.isupper():  # White
            piece = letter_to_piece[white][c]
            pos.pieces[white][piece] = set_bit(pos.pieces[white][piece], squares[sq_i])
            sq_i += 1

        elif c.islower():  # Black
            piece = letter_to_piece[black][c]
            pos.pieces[black][piece] = set_bit(pos.pieces[black][piece], squares[sq_i])
            sq_i += 1

        elif c.isnumeric():  # Empty
            sq_i += int(c)

    for i in range(2):
        for bb in pos.pieces[i]:
            pos.occupancy[i] |= bb

    pos.occupancy[both] = pos.occupancy[white] | pos.occupancy[black]

    return pos

