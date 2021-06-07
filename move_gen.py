from constants import *
from bb_operations import *
from attack_tables import *
from position import Position

"""
           Binary move bits             Meaning          Hexadecimal

    0000 0000 0000 0000 0011 1111    source square       0x3f
    0000 0000 0000 1111 1100 0000    target square       0xfc0
    0000 0000 0111 0000 0000 0000    piece               0x7000
    0000 0000 1000 0000 0000 0000    side                0x8000
    0000 1111 0000 0000 0000 0000    promoted piece      0xf0000
    0001 0000 0000 0000 0000 0000    capture flag        0x100000
    0010 0000 0000 0000 0000 0000    double push flag    0x200000
    0100 0000 0000 0000 0000 0000    enpassant flag      0x400000
    1000 0000 0000 0000 0000 0000    castling flag       0x800000

"""


@njit
def encode_move(source, target, piece, side, promote_to, capture, double, enpas, castling):
    return source | target << 6 | piece << 12 | side << 15 | promote_to << 16 | capture << 20 | \
           double << 21 | enpas << 22 | castling << 23


@njit
def get_move_source(move):
    return move & 0x3f


@njit
def get_move_target(move):
    return (move & 0xfc0) >> 6


@njit
def get_move_piece(move):
    return (move & 0x7000) >> 12


@njit
def get_move_side(move) -> (1, 2):
    return int(bool(move & 0x8000))


@njit
def get_move_promote_to(move):
    return (move & 0xf0000) >> 16


@njit
def get_move_capture(move):
    return move & 0x100000


@njit
def get_move_double(move):
    return move & 0x200000


@njit
def get_move_enpas(move):
    return move & 0x400000


@njit
def get_move_castling(move):
    return move & 0x800000


def get_move_uci(move):
    return f"{square_to_coordinates[get_move_source(move)]}{square_to_coordinates[get_move_target(move)]}" \
           f"{promo_piece_to_str[get_move_promote_to(move)] if get_move_promote_to(move) else ''}"


def print_move(move):
    """print a move in UCI format"""
    print(f"{square_to_coordinates[get_move_source(move)]}{square_to_coordinates[get_move_target(move)]}"
          f"{promo_piece_to_str[get_move_promote_to(move)] if get_move_promote_to(move) else ''}")


def print_move_list(move_list):
    """print a nice move list"""
    if not move_list:
        print("Empty move_list")

    print()
    print("  move    piece    capture    double    enpas    castling")

    for move in move_list:
        print(f"  {square_to_coordinates[get_move_source(move)]}{square_to_coordinates[get_move_target(move)]}"
              f"{promo_piece_to_str[get_move_promote_to(move)] if get_move_promote_to(move) else ''}     "
              f"{piece_to_ascii[int(bool(get_move_side(move)))][get_move_piece(move)]}         "
              f"{int(bool(get_move_capture(move)))}         {int(bool(get_move_double(move)))}         "
              f"{int(bool(get_move_enpas(move)))}         "
              f"{int(bool(get_move_castling(move)))}")

    print("Total number of moves:", len(move_list))


@njit
def is_square_attacked(pos, sq, side):
    """is side attacking sq"""
    opp = black if not side else white
    if pawn_attacks[opp][sq] & pos.pieces[side][pawn] \
            or knight_attacks[sq] & pos.pieces[side][knight] \
            or get_bishop_attacks(sq, pos.occupancy[both]) & pos.pieces[side][bishop] \
            or get_rook_attacks(sq, pos.occupancy[both]) & pos.pieces[side][rook] \
            or get_queen_attacks(sq, pos.occupancy[both]) & pos.pieces[side][queen] \
            or king_attacks[sq] & pos.pieces[side][king]:
        return True
    return False


def print_attacked_square(pos, side):
    """print a bitboard of all squares attacked by a given side"""
    attacked = EMPTY
    for sq in squares:
        if is_square_attacked(pos, sq, side):
            attacked = set_bit(attacked, sq)
    print_bb(attacked)


@njit
def generate_moves(pos):
    """return a list of pseudo legal moves from a given Position"""

    # TODO: integrate the constants to be able to compile AOT

    move_list = []

    for piece in range(6):
        bb = pos.pieces[pos.side][piece]
        opp = white if pos.side else black

        # white pawns & king castling moves
        if pos.side == white:
            if piece == pawn:
                while bb:
                    # pawn move
                    source = np.uint8(get_ls1b_index(bb))
                    target = source - np.uint8(8)

                    # quiet pawn move
                    if not target < a8 and not get_bit(pos.occupancy[both], target):

                        # promotion
                        if a7 <= source <= h7:
                            move_list.append(encode_move(source, target, piece, pos.side, queen, 0, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, rook, 0, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, bishop, 0, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, knight, 0, 0, 0, 0))

                        else:
                            # push
                            move_list.append(encode_move(source, target, piece, pos.side, 0, 0, 0, 0, 0))

                            # push push
                            if a2 <= source <= h2 and not get_bit(pos.occupancy[both], target - np.uint8(8)):
                                move_list.append(
                                    encode_move(source, target - np.uint8(8), piece, pos.side, 0, 0, 1, 0, 0))

                    # pawn attack tables
                    attacks = pawn_attacks[white][source] & pos.occupancy[black]

                    while attacks:
                        target = get_ls1b_index(attacks)

                        # promotion capture
                        if a7 <= source <= h7:
                            move_list.append(encode_move(source, target, piece, pos.side, queen, 1, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, rook, 1, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, bishop, 1, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, knight, 1, 0, 0, 0))
                        # capture
                        else:
                            move_list.append(encode_move(source, target, piece, pos.side, 0, 1, 0, 0, 0))

                        attacks = pop_bit(attacks, target)

                    # en-passant
                    if pos.enpas != no_sq:
                        enpas_attacks = pawn_attacks[white][source] & (BIT << pos.enpas)

                        if enpas_attacks:
                            target_enpas = get_ls1b_index(enpas_attacks)
                            move_list.append(encode_move(source, target_enpas, piece, pos.side, 0, 1, 0, 1, 0))

                    bb = pop_bit(bb, source)

            if piece == king:
                if pos.castle & wk:
                    # are squares empty
                    if not get_bit(pos.occupancy[both], f1) and not get_bit(pos.occupancy[both], g1):
                        # are squares safe
                        if not is_square_attacked(pos, e1, black) and not is_square_attacked(pos, f1, black):
                            move_list.append(encode_move(e1, g1, piece, pos.side, 0, 0, 0, 0, 1))

                if pos.castle & wq:
                    # squares are empty
                    if not get_bit(pos.occupancy[both], d1) and not get_bit(pos.occupancy[both], c1) and not get_bit(
                            pos.occupancy[both], b1):
                        # squares are not attacked by black
                        if not is_square_attacked(pos, e1, black) and not is_square_attacked(pos, d1, black):
                            move_list.append(encode_move(e1, c1, piece, pos.side, 0, 0, 0, 0, 1))

        # black pawns & king castling moves
        if pos.side == black:
            if piece == pawn:
                while bb:
                    source = get_ls1b_index(bb)
                    target = source + np.uint8(8)

                    # quiet pawn move
                    if not target > h1 and not get_bit(pos.occupancy[both], target):

                        # Promotion
                        if a2 <= source <= h2:
                            move_list.append(encode_move(source, target, piece, pos.side, queen, 0, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, rook, 0, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, bishop, 0, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, knight, 0, 0, 0, 0))

                        else:
                            # push
                            move_list.append(encode_move(source, target, piece, pos.side, 0, 0, 0, 0, 0))

                            # push push
                            if a7 <= source <= h7 and not get_bit(pos.occupancy[both], target + np.uint8(8)):
                                move_list.append(
                                    encode_move(source, target + np.uint8(8), piece, pos.side, 0, 0, 1, 0, 0))

                    # pawn attack tables
                    attacks = pawn_attacks[black][source] & pos.occupancy[white]

                    while attacks:
                        target = get_ls1b_index(attacks)

                        # promotion capture
                        if a2 <= source <= h2:
                            move_list.append(encode_move(source, target, piece, pos.side, queen, 1, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, rook, 1, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, bishop, 1, 0, 0, 0))
                            move_list.append(encode_move(source, target, piece, pos.side, knight, 1, 0, 0, 0))
                        # capture
                        else:
                            move_list.append(encode_move(source, target, piece, pos.side, 0, 1, 0, 0, 0))

                        attacks = pop_bit(attacks, target)

                    # en-passant
                    if pos.enpas != no_sq:
                        enpas_attacks = pawn_attacks[black][source] & (BIT << pos.enpas)

                        if enpas_attacks:
                            target_enpas = get_ls1b_index(enpas_attacks)
                            move_list.append(encode_move(source, target_enpas, piece, pos.side, 0, 1, 0, 1, 0))

                    bb = pop_bit(bb, source)

            if piece == king:  # target square will be checked later with legality
                if pos.castle & bk:
                    # squares are empty
                    if not get_bit(pos.occupancy[both], f8) and not get_bit(pos.occupancy[both], g8):
                        # squares are not attacked by black
                        if not is_square_attacked(pos, e8, white) and not is_square_attacked(pos, f8, white):
                            move_list.append(encode_move(e8, g8, piece, pos.side, 0, 0, 0, 0, 1))

                if pos.castle & bq:
                    # squares are empty
                    if not get_bit(pos.occupancy[both], d8) and not get_bit(pos.occupancy[both], c8) and not get_bit(
                            pos.occupancy[both], b8):
                        # squares are not attacked by white
                        if not is_square_attacked(pos, e8, white) and not is_square_attacked(pos, d8, white):
                            move_list.append(encode_move(e8, c8, piece, pos.side, 0, 0, 0, 0, 1))

        if piece in range(1, 6):
            while bb:
                source = np.uint8(get_ls1b_index(bb))
                attacks = get_attacks(piece, source, pos)

                while attacks != EMPTY:
                    target = np.uint8(get_ls1b_index(attacks))

                    # quiet
                    if not get_bit(pos.occupancy[opp], target):
                        move_list.append(encode_move(source, target, piece, pos.side, 0, 0, 0, 0, 0))

                    # capture
                    else:
                        move_list.append(encode_move(source, target, piece, pos.side, 0, 1, 0, 0, 0))

                    attacks = pop_bit(attacks, target)

                bb = pop_bit(bb, source)

    return move_list


def generate_legal_moves(pos):
    """very inefficient, use only to debug"""
    return [move for move in generate_moves(pos) if make_move(pos, move)]


@njit
def make_move(pos_orig, move, only_captures=0):
    """return new updated position if (move is legal) else None"""

    # TODO: integrate the constants to be able to compile AOT

    # create a copy of the position
    pos = Position()
    pos.pieces = pos_orig.pieces.copy()
    pos.occupancy = pos_orig.occupancy.copy()
    pos.side = pos_orig.side
    pos.enpas = pos_orig.enpas
    pos.castle = pos_orig.castle

    # quiet moves
    if not only_captures:

        # parse move
        source_square = np.uint8(get_move_source(move))
        target_square = np.uint8(get_move_target(move))
        piece = get_move_piece(move)
        side = get_move_side(move)
        opp = white if side else black
        promote_to = get_move_promote_to(move)
        capture = get_move_capture(move)
        double_push = get_move_double(move)
        enpas = get_move_enpas(move)
        castling = get_move_castling(move)

        pos.pieces[side][piece] = pop_bit(pos.pieces[side][piece], source_square)
        pos.pieces[side][piece] = set_bit(pos.pieces[side][piece], target_square)

        if capture:  # find what we captured and erase it
            for opp_piece in range(6):
                if get_bit(pos.pieces[opp][opp_piece], target_square):
                    pos.pieces[opp][opp_piece] = pop_bit(pos.pieces[opp][opp_piece], target_square)
                    break

        if promote_to:  # erase pawn and place promoted piece
            pos.pieces[side][piece] = pop_bit(pos.pieces[side][piece], target_square)
            pos.pieces[side][promote_to] = set_bit(pos.pieces[side][promote_to], target_square)

        if enpas:  # erase the opp pawn
            if side:  # black just moved
                pos.pieces[opp][piece] = pop_bit(pos.pieces[opp][piece], target_square - 8)
            else:  # white just moved
                pos.pieces[opp][piece] = pop_bit(pos.pieces[opp][piece], target_square + 8)

        # reset enpas
        pos.enpas = no_sq

        if double_push:  # set en-passant square
            if side:  # black just moved
                pos.enpas = target_square - np.uint8(8)
            else:  # white just moved
                pos.enpas = target_square + np.uint8(8)

        if castling:  # move rook accordingly
            if target_square == g1:
                pos.pieces[side][rook] = pop_bit(pos.pieces[side][rook], h1)
                pos.pieces[side][rook] = set_bit(pos.pieces[side][rook], f1)
            elif target_square == c1:
                pos.pieces[side][rook] = pop_bit(pos.pieces[side][rook], a1)
                pos.pieces[side][rook] = set_bit(pos.pieces[side][rook], d1)
            elif target_square == g8:
                pos.pieces[side][rook] = pop_bit(pos.pieces[side][rook], h8)
                pos.pieces[side][rook] = set_bit(pos.pieces[side][rook], f8)
            elif target_square == c8:
                pos.pieces[side][rook] = pop_bit(pos.pieces[side][rook], a8)
                pos.pieces[side][rook] = set_bit(pos.pieces[side][rook], d8)

        # update castling rights
        pos.castle &= castling_rights[source_square]
        pos.castle &= castling_rights[target_square]

        # clear occupancy
        pos.occupancy = np.zeros(3, dtype=np.uint64)

        # update occupancy
        for color in (white, black):
            for bb in pos.pieces[color]:
                pos.occupancy[color] |= bb
            pos.occupancy[both] |= pos.occupancy[color]

        pos.side = opp

        if is_square_attacked(pos, get_ls1b_index(pos.pieces[side][king]), opp):
            return None

        else:
            return pos

    return

    # # Capturing moves
    # else:
    #     if get_move_capture(move):
    #         return make_move(pos, move, only_captures=False)
    #
    #     else:
    #         return 0
