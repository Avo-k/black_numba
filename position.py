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
        self.enpas = no_sq
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
        print("en passant:", square_to_coordinates[pos.enpas])
        casl = f"{'K' if pos.castle & wk else ''}{'Q' if pos.castle & wq else ''}" \
               f"{'k' if pos.castle & bk else ''}{'q' if pos.castle & bq else ''} "
        print("Castling:",casl if casl else "-", "\n")





# @njit
def parse_fen(fen: str):
    """return a Position object from a Forsyth-Edwards Notation string"""

    pos = Position()

    # numba dict helper
    num_str_to_int = nb.typed.Dict.empty(nb.types.string, nb.types.int64)
    for num in range(1, 9):
        num_str_to_int[str(num)] = num

    let_str_to_int = nb.typed.Dict.empty(nb.types.string, nb.types.int64)
    for side in (('P', 'N', 'B', 'R', 'Q', 'K'), ('p', 'n', 'b', 'r', 'q', 'k')):
        for code, letter in enumerate(side):
            let_str_to_int[letter] = code

    squar_to_coordinates = [
        "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
        "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
        "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
        "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
        "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
        "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
        "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
        "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "-"]

    board, color, castle, ep, _hclock, _fclock = fen.split()

    pos.side = 0 if color == "w" else 1

    if ep == "-":
        pos.enpas = no_sq
    else:
        for i, sq in enumerate(squar_to_coordinates):
            if sq == ep:
                pos.enpas = i

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
            piece = let_str_to_int[c]
            pos.pieces[white][piece] = set_bit(pos.pieces[white][piece], squares[sq_i])
            sq_i += 1

        elif c.islower():  # Black
            piece = let_str_to_int[c]
            pos.pieces[black][piece] = set_bit(pos.pieces[black][piece], squares[sq_i])
            sq_i += 1

        elif c.isnumeric():  # Empty
            sq_i += num_str_to_int[c]

    for i in range(2):
        for bb in pos.pieces[i]:
            pos.occupancy[i] |= bb

    pos.occupancy[both] = pos.occupancy[white] | pos.occupancy[black]

    return pos

