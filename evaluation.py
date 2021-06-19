from constants import *
from bb_operations import *
from position import Position
from attack_tables import get_bishop_attacks, get_queen_attacks, king_attacks, knight_attacks


@njit(nb.int64(Position.class_type.instance_type))
def evaluate(pos) -> int:
    """return evaluation of a position from side-to-play perspective"""
    score = 0

    # Doubled pawn
    num_white_double_pawn = count_bits(pos.pieces[0][pawn] & (pos.pieces[0][pawn] << 8))
    num_black_double_pawn = count_bits(pos.pieces[1][pawn] & (pos.pieces[1][pawn] << 8))
    score += num_white_double_pawn * double_pawn_penalty - num_black_double_pawn * double_pawn_penalty

    for color in range(2):
        for piece in range(6):

            bb = pos.pieces[color][piece]
            pst = PST[piece]

            while bb:
                sq = get_ls1b_index(bb)

                if color:  # black

                    # Material score
                    score -= material_score[piece]

                    # Positional score
                    score -= pst[mirror_pst[sq]]

                    if piece == pawn:
                        # Isolated pawn
                        if not pos.pieces[color][piece] & isolated_masks[sq]:
                            score -= isolated_pawn_penalty
                        # Passed pawn
                        if not black_passed_masks[sq] & pos.pieces[color ^ 1][piece]:
                            score -= passed_pawn_bonus[mirror_pst[sq] // 8]

                    elif piece == knight:
                        moves = count_bits(knight_attacks[sq] & ~pos.occupancy[color])
                        score -= (moves - 4) * 4

                    elif piece == bishop:
                        # Mobility
                        moves = count_bits(get_bishop_attacks(sq, pos.occupancy[both]))
                        score -= (moves - 6) * 5

                    elif piece == rook:
                        # Semi-open file
                        if not pos.pieces[color][pawn] & file_masks[sq]:
                            score -= semi_open_file_bonus
                            # Open file
                            if not (pos.pieces[color][pawn] | pos.pieces[color ^ 1][pawn]) & file_masks[sq]:
                                score -= open_file_bonus

                    elif piece == queen:
                        # Mobility
                        moves = count_bits(get_queen_attacks(sq, pos.occupancy[both]))
                        score -= (moves - 10) * 3

                    elif piece == king:
                        # Semi-open file
                        if not pos.pieces[color][pawn] & file_masks[sq]:
                            score += semi_open_file_bonus
                            # Open file
                            if not (pos.pieces[color][pawn] | pos.pieces[color ^ 1][pawn]) & file_masks[sq]:
                                score += open_file_bonus
                        # King safety
                        score -= count_bits(king_attacks[sq] & pos.occupancy[color]) * king_shield_bonus

                else:  # white

                    # Material score
                    score += material_score[piece]

                    # Positional score
                    score += pst[sq]

                    if piece == pawn:
                        # Isolated pawn
                        if not pos.pieces[color][piece] & isolated_masks[sq]:
                            score += isolated_pawn_penalty
                        # Passed pawn
                        if not white_passed_masks[sq] & pos.pieces[color ^ 1][piece]:
                            score += passed_pawn_bonus[sq // 8]

                    elif piece == knight:
                        moves = count_bits(knight_attacks[sq] & ~pos.occupancy[color])
                        score += (moves - 4) * 4

                    elif piece == bishop:
                        # Mobility
                        moves = count_bits(get_bishop_attacks(sq, pos.occupancy[both]))
                        score += (moves - 6) * 5

                    elif piece == rook:
                        # Semi-open file
                        if not pos.pieces[color][pawn] & file_masks[sq]:
                            score += semi_open_file_bonus
                            # Open file
                            if not (pos.pieces[color][pawn] | pos.pieces[color ^ 1][pawn]) & file_masks[sq]:
                                score += open_file_bonus

                    elif piece == queen:
                        # Mobility
                        moves = count_bits(get_queen_attacks(sq, pos.occupancy[both]))
                        score += (moves - 10) * 3

                    elif piece == king:
                        # Semi-open file
                        if not pos.pieces[color][pawn] & file_masks[sq]:
                            score -= semi_open_file_bonus
                            # Open file
                            if not (pos.pieces[color][pawn] | pos.pieces[color ^ 1][pawn]) & file_masks[sq]:
                                score -= open_file_bonus
                        # King safety
                        score += count_bits(king_attacks[sq] & pos.occupancy[color]) * king_shield_bonus

                bb = pop_bit(bb, sq)

    # score from white perspective
    return -score if pos.side else score
