from constants import *
from bb_operations import *
from position import Position
from attack_tables import get_bishop_attacks, get_queen_attacks, king_attacks, knight_attacks


@njit
def get_game_phase_score(pos):
    score = 0
    for color in (black, white):
        for piece in range(knight, king):
            score += count_bits(pos.pieces[color][piece]) * material_score[piece]
    return score


@njit(cache=True)
def king_mg(pos, sq, color):
    v = 0
    # Semi-open file
    if not pos.pieces[color][pawn] & file_masks[sq]:
        v -= semi_open_file_bonus
        # Open file
        if not (pos.pieces[color][pawn] | pos.pieces[color ^ 1][pawn]) & file_masks[sq]:
            v -= open_file_bonus
    # King safety
    v += count_bits(king_attacks[sq] & pos.occupancy[color]) * king_shield_bonus
    return v


@njit()
def queen_mg(pos, sq):
    """return queen middle-game evaluation"""
    # mobility (-19 to 19)
    moves = count_bits(get_queen_attacks(sq, pos.occupancy[both]))
    return moves - 5


@njit(cache=True)
def rook_mg(pos, sq, color):
    v = 0
    # Semi-open file
    if not pos.pieces[color][pawn] & file_masks[sq]:
        v += semi_open_file_bonus
        # Open file
        if not (pos.pieces[color][pawn] | pos.pieces[color ^ 1][pawn]) & file_masks[sq]:
            v += open_file_bonus
    # Mobility (-28 to 28)
    moves = count_bits(get_bishop_attacks(sq, pos.occupancy[both]))
    v += (moves - 6) * 2
    return v


@njit(cache=True)
def bishop_mg(pos, sq):
    # Mobility (-30 to 30)
    moves = count_bits(get_bishop_attacks(sq, pos.occupancy[both]))
    return (moves - 3) * 5


@njit(cache=True)
def knight_mg(pos, sq, color):
    # Mobility (from -16 to 16)
    moves = count_bits(knight_attacks[sq] & ~pos.occupancy[color])
    return (moves - 4) * 4


@njit(cache=True)
def pawn_mg(pos, sq, color):
    v = 0
    if color:
        # Isolated pawn
        if not pos.pieces[color][pawn] & isolated_masks[sq]:
            v += isolated_pawn_penalty
        # Passed pawn
        if not black_passed_masks[sq] & pos.pieces[color ^ 1][pawn]:
            v += passed_pawn_bonus[mirror_pst[sq] // 8]
    else:
        # Isolated pawn
        if not pos.pieces[color][pawn] & isolated_masks[sq]:
            v += isolated_pawn_penalty
        # Passed pawn
        if not white_passed_masks[sq] & pos.pieces[color ^ 1][pawn]:
            v += passed_pawn_bonus[sq // 8]
    return v


@njit(nb.int64(Position.class_type.instance_type))
def evaluate(pos) -> int:
    """return evaluation of a position from side-to-play perspective"""
    score = 0

    for color in (black, white):
        double_pawns = count_bits(pos.pieces[color][pawn] & (pos.pieces[color][pawn] << 8))
        score += double_pawns * double_pawn_penalty

        for piece in range(6):
            bb = pos.pieces[color][piece]
            pst = PST[piece]

            while bb:
                sq = get_ls1b_index(bb)

                # Material score
                score += material_score[piece]

                # Positional score
                score += pst[sq]

                if piece == king:
                    score += king_mg(pos, sq, color)

                elif piece == pawn:
                    score += pawn_mg(pos, sq, color)

                elif piece == rook:
                    score += rook_mg(pos, sq, color)

                elif piece == queen:
                    score += queen_mg(pos, sq)

                elif piece == knight:
                    score += knight_mg(pos, sq, color)

                elif piece == bishop:
                    score += bishop_mg(pos, sq)

                bb = pop_bit(bb, sq)

        if color:
            score = -score

    return -score if pos.side else score
