from constants import *
from bb_operations import *
from numba.experimental import jitclass

position_spec = [
    ('pieces', nb.uint64[:, :]),
    ('occupancy', nb.uint64[:]),
    ('side', nb.uint8),
    ('enpas', nb.uint8),
    ('castle', nb.uint8),
    ('hash_key', nb.uint64)]


@jitclass(position_spec)
class Position:
    def __init__(self):
        self.pieces = np.zeros((2, 6), dtype=np.uint64)  # bb for each color (2) and each piece type (6)
        self.occupancy = np.zeros(3, dtype=np.uint64)  # Combined bitboards for (white, black, both)
        self.side = 0
        self.enpas = no_sq
        self.castle = 0
        self.hash_key = 0


def print_position(pos, print_info=False):
    print("\n")
    for rank in range(8):
        r = " +---+---+---+---+---+---+---+---+\n |"
        for file in range(8):
            sq = np.uint8(rank * 8 + file)

            for side in range(2):
                if get_bit(pos.occupancy[side], sq):
                    for piece, bb in enumerate(pos.pieces[side]):
                        if get_bit(bb, sq):
                            r += f" {piece_to_letter[side][piece]} |"
                            break
                    break

            # empty square
            else:
                r += "   |"

        # assert len(r) == 25
        r += f" {8 - rank}"
        print(r)
    print(" +---+---+---+---+---+---+---+---+")
    print("   A   B   C   D   E   F   G   H\n")

    if print_info:
        print("white" if not pos.side else "black", "to move")
        print("en passant:", square_to_coordinates[pos.enpas])
        casl = f"{'K' if pos.castle & wk else ''}{'Q' if pos.castle & wq else ''}" \
               f"{'k' if pos.castle & bk else ''}{'q' if pos.castle & bq else ''} "
        print("Castling:", casl if casl else "-")
        print("Hash key:", hex(pos.hash_key), "\n")


@njit(nb.uint64(Position.class_type.instance_type))
def generate_hash_key(pos):
    """generate a hash_key from a position"""

    final_key = 0

    for color in range(2):
        for piece in range(6):
            bb = pos.pieces[color][piece]
            while bb:
                square = get_ls1b_index(bb)

                final_key ^= piece_keys[color][piece][square]

                bb = pop_bit(bb, square)

    # todo: get rid of it by having 65 sq array
    if pos.enpas != no_sq:
        final_key ^= en_passant_keys[pos.enpas]

    final_key ^= castle_keys[pos.castle]

    if pos.side:
        final_key ^= side_key

    return final_key


@njit(Position.class_type.instance_type(nb.types.string))
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

    board, color, castle, ep, _hclock, _fclock = fen.split()

    pos.side = 0 if color == "w" else 1

    if ep == "-":
        pos.enpas = no_sq
    else:
        for i, sq in enumerate(square_to_coordinates):
            if sq == ep:
                pos.enpas = i
                break

    pos.castle = 0
    for i, c in enumerate("KQkq"):
        if c in castle:
            pos.castle += (2 ** i)

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

    pos.hash_key = generate_hash_key(pos)

    return pos
