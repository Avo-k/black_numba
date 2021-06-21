from constants import *
from bb_operations import *
from position import Position
from attack_tables import get_bishop_attacks, get_queen_attacks, king_attacks, knight_attacks, get_rook_attacks


@njit
def get_game_phase_score(pos):
    score = 0
    for color in (black, white):
        for piece in range(knight, king):
            score += count_bits(pos.pieces[color][piece]) * material_score[piece]
    return score


def king_safety(pos, sq, color):
    v = 0
    # King shield
    v += -15 + count_bits(king_attacks[sq] & pos.occupancy[color]) * king_shield_bonus

    # # Pawn tropism
    # p_tropism = 0
    # nb_of_pawn = 0
    # bb = pos.piece[color][pawn]
    # kg_sq = get_ls1b_index(pos.piece[color][king])
    #
    # while bb:
    #     sq = get_ls1b_index(bb)
    #     p_tropism += arr_manhattan[kg_sq][sq]
    #     nb_of_pawn += 1
    #     bb = pop_bit(bb, sq)
    #
    # bb_opp = pos.piece[color ^ 1][pawn]


@njit(nb.int64(Position.class_type.instance_type))
def evaluate(pos) -> int:
    """return evaluation of a position from side-to-play perspective"""
    score = 0

    # tropism (white, black)
    tropism = np.zeros(2, dtype=np.uint8)

    wk_sq = get_ls1b_index(pos.pieces[white][king])
    bk_sq = get_ls1b_index(pos.pieces[black][king])
    king_sq = (wk_sq, bk_sq)

    for color in (black, white):

        opp = color ^ 1

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
                    # Semi-open file
                    if not pos.pieces[color][pawn] & file_masks[sq]:
                        score -= semi_open_file_bonus
                        # Open file
                        if not (pos.pieces[color][pawn] | pos.pieces[opp][pawn]) & file_masks[sq]:
                            score -= open_file_bonus

                elif piece == pawn:
                    if color:
                        # Isolated pawn
                        if not pos.pieces[color][pawn] & isolated_masks[sq]:
                            score += isolated_pawn_penalty
                        # Passed pawn
                        if not black_passed_masks[sq] & pos.pieces[opp][pawn]:
                            score += passed_pawn_bonus[mirror_pst[sq] // 8]
                    else:
                        # Isolated pawn
                        if not pos.pieces[color][pawn] & isolated_masks[sq]:
                            score += isolated_pawn_penalty
                        # Passed pawn
                        if not white_passed_masks[sq] & pos.pieces[opp][pawn]:
                            score += passed_pawn_bonus[sq // 8]

                elif piece == rook:
                    # Semi-open file
                    if not pos.pieces[color][pawn] & file_masks[sq]:
                        score += semi_open_file_bonus
                        # Open file
                        if not (pos.pieces[color][pawn] | pos.pieces[opp][pawn]) & file_masks[sq]:
                            score += open_file_bonus
                    # Mobility (-28 to 28)
                    moves = count_bits(get_rook_attacks(sq, pos.occupancy[both]))
                    score += (moves - 6) * 2
                    # Tropism
                    tropism[opp] += arr_manhattan[sq][king_sq[color]] * rook_tropism_mg

                elif piece == queen:
                    # mobility (-19 to 19)
                    moves = count_bits(get_queen_attacks(sq, pos.occupancy[both]))
                    score += moves - 5
                    # Tropism
                    tropism[opp] += arr_manhattan[sq][king_sq[color]] * queen_tropism_mg

                elif piece == knight:
                    # Mobility (from -16 to 16)
                    moves = count_bits(knight_attacks[sq] & ~pos.occupancy[color])
                    score += (moves - 4) * 4
                    # Tropism
                    tropism[opp] += arr_manhattan[sq][king_sq[color]] * knight_tropism_mg

                elif piece == bishop:
                    # Mobility (-30 to 30)
                    moves = count_bits(get_bishop_attacks(sq, pos.occupancy[both]))
                    score += (moves - 3) * 5
                    # Tropism
                    tropism[opp] += arr_manhattan[sq][king_sq[color]] * bishop_tropism_mg

                bb = pop_bit(bb, sq)

        if color:
            score = -score

    score -= tropism[0]
    score += tropism[1]

    return -score if pos.side else score
