from constants import *
from bb_operations import *
from position import Position
from attack_tables import get_bishop_attacks, get_queen_attacks, king_attacks, knight_attacks, get_rook_attacks
from pst import PST


@njit(nb.uint16(Position.class_type.instance_type), cache=True)
def get_game_phase_score(pos):
    score = 0
    for color in (black, white):
        for piece in range(knight, king):
            score += count_bits(pos.pieces[color][piece]) * phase_scores[piece]
    return (score * 256 + (TOTAL_PHASE / 2)) / TOTAL_PHASE


@njit(cache=True)
def bishop_mg(pos, sq, kings_sq, opp):
    v = 0
    # Mobility (-30 to 30)
    moves = count_bits(get_bishop_attacks(sq, pos.occupancy[both]))
    v += (moves - 3) * 5
    # Tropism
    v += bishop_tropism_mg * 8 - arr_manhattan[sq][kings_sq[opp]] * bishop_tropism_mg
    return v


@njit(cache=True)
def bishop_eg(pos, sq, kings_sq, opp):
    v = 0
    # Mobility (-30 to 30)
    moves = count_bits(get_bishop_attacks(sq, pos.occupancy[both]))
    v += (moves - 3) * 5
    # Tropism
    v += bishop_tropism_eg * 8 - arr_manhattan[sq][kings_sq[opp]] * bishop_tropism_eg
    return v


@njit(cache=True)
def knight_mg(pos, sq, kings_sq, opp, color):
    v = 0
    # Mobility (from -16 to 16)
    moves = count_bits(knight_attacks[sq] & ~pos.occupancy[color])
    v += (moves - 4) * 4
    # Tropism
    v += knight_tropism_mg * 8 - arr_manhattan[sq][kings_sq[opp]] * knight_tropism_mg
    return v


@njit(cache=True)
def knight_eg(pos, sq, kings_sq, opp, color):
    v = 0
    # Mobility (from -16 to 16)
    moves = count_bits(knight_attacks[sq] & ~pos.occupancy[color])
    v += (moves - 4) * 4
    # Tropism
    v += knight_tropism_eg * 8 - arr_manhattan[sq][kings_sq[opp]] * knight_tropism_eg
    return v


@njit
def queen_mg(pos, sq, kings_sq, opp):
    v = 0
    # mobility (-19 to 19)
    moves = count_bits(get_queen_attacks(sq, pos.occupancy[both]))
    v += moves - 5
    # Tropism
    v += queen_tropism_mg * 8 - arr_manhattan[sq][kings_sq[opp]] * queen_tropism_mg
    return v


@njit
def queen_eg(pos, sq, kings_sq, opp):
    v = 0
    # mobility (-19 to 19)
    moves = count_bits(get_queen_attacks(sq, pos.occupancy[both]))
    v += (moves - 5) * 2
    # Tropism
    v += queen_tropism_eg * 8 - arr_manhattan[sq][kings_sq[opp]] * queen_tropism_eg
    return v


@njit
def rook_mg(pos, sq, kings_sq, opp, color):
    v = 0
    # Semi-open file
    if not pos.pieces[color][pawn] & file_masks[sq]:
        v += semi_open_file_bonus
        # Open file
        if not (pos.pieces[color][pawn] | pos.pieces[opp][pawn]) & file_masks[sq]:
            v += open_file_bonus
    # Mobility (-28 to 28)
    moves = count_bits(get_rook_attacks(sq, pos.occupancy[both]))
    v += (moves - 6) * 2
    # Tropism
    v += rook_tropism_mg * 8 - arr_manhattan[sq][kings_sq[opp]] * rook_tropism_mg
    return v


@njit
def rook_eg(pos, sq, kings_sq, opp, color):
    v = 0
    # Semi-open file
    if not pos.pieces[color][pawn] & file_masks[sq]:
        v += semi_open_file_bonus
        # Open file
        if not (pos.pieces[color][pawn] | pos.pieces[opp][pawn]) & file_masks[sq]:
            v += open_file_bonus
    # Mobility (-28 to 28)
    moves = count_bits(get_rook_attacks(sq, pos.occupancy[both]))
    v += (moves - 6) * 4
    # Tropism
    v += rook_tropism_eg * 8 - arr_manhattan[sq][kings_sq[opp]] * rook_tropism_eg
    return v


@njit(cache=True)
def pawn_mg(pos, sq, opp, color):
    v = 0
    if color:
        # Isolated pawn
        if not pos.pieces[color][pawn] & isolated_masks[sq]:
            v += isolated_pawn_penalty
        # Passed pawn
        if not black_passed_masks[sq] & pos.pieces[opp][pawn]:
            v += passed_pawn_bonus[mirror_pst[sq] // 8]
    else:
        # Isolated pawn
        if not pos.pieces[color][pawn] & isolated_masks[sq]:
            v += isolated_pawn_penalty
        # Passed pawn
        if not white_passed_masks[sq] & pos.pieces[opp][pawn]:
            v += passed_pawn_bonus[sq // 8]
    return v


@njit
def king_mg(pos, sq, opp, color):
    v = 0
    # Semi-open file
    if not pos.pieces[color][pawn] & file_masks[sq]:
        v -= semi_open_file_bonus
        # Open file
        if not (pos.pieces[color][pawn] | pos.pieces[opp][pawn]) & file_masks[sq]:
            v -= open_file_bonus

    # Pawn shield
    v += -15 + count_bits(king_attacks[sq] & pos.occupancy[color]) * king_shield_bonus

    # "Anti-mobility"
    moves = count_bits(get_queen_attacks(sq, pos.occupancy[both]))
    v -= (moves - 5) * 2
    return v


@njit
def king_eg(pos, sq, opp, color):
    v = 0
    # Pawn shield
    v += count_bits(king_attacks[sq] & pos.occupancy[color]) * king_shield_bonus

    # "Anti-mobility"
    moves = count_bits(get_queen_attacks(sq, pos.occupancy[both]))
    v -= moves - 8
    return v


@njit(nb.int64(Position.class_type.instance_type))
def evaluate(pos) -> int:
    """return evaluation of a position from side-to-play perspective"""
    mg_score = 0
    eg_score = 0

    game_phase_score = get_game_phase_score(pos)
    kings_sq = (get_ls1b_index(pos.pieces[white][king]), get_ls1b_index(pos.pieces[black][king]))

    for color in (black, white):

        opp = color ^ 1

        # double pawns
        double_pawns = count_bits(pos.pieces[color][pawn] & (pos.pieces[color][pawn] << 8))
        mg_score += double_pawns * double_pawn_penalty
        # bishop counter
        bish = 0

        for piece in range(6):
            bb = pos.pieces[color][piece]

            while bb:
                sq = get_ls1b_index(bb)

                # Material score
                mg_score += material_scores[opening][piece]
                eg_score += material_scores[end_game][piece]

                # Positional score
                mg_score += PST[opening][piece][sq]
                eg_score += PST[end_game][piece][sq]

                if piece == king:
                    mg_score += king_mg(pos, sq, opp, color)
                    eg_score += king_eg(pos, sq, opp, color)

                elif piece == pawn:
                    mg_score += pawn_mg(pos, sq, opp, color)
                    eg_score += pawn_mg(pos, sq, opp, color)

                elif piece == rook:
                    mg_score += rook_mg(pos, sq, kings_sq, opp, color)
                    eg_score += rook_eg(pos, sq, kings_sq, opp, color)

                elif piece == queen:
                    mg_score += queen_mg(pos, sq, kings_sq, opp)
                    eg_score += queen_eg(pos, sq, kings_sq, opp)

                elif piece == knight:
                    mg_score += knight_mg(pos, sq, kings_sq, opp, color)
                    eg_score += knight_eg(pos, sq, kings_sq, opp, color)

                elif piece == bishop:
                    bish += 1
                    mg_score += bishop_mg(pos, sq, kings_sq, opp)
                    eg_score += bishop_eg(pos, sq, kings_sq, opp)

                bb = pop_bit(bb, sq)

        if bish > 1:
            mg_score += bishop_pair_mg
            eg_score += bishop_pair_eg

        if color:
            mg_score, eg_score = -mg_score, -eg_score

    # Initiative
    if game_phase_score > 50:
        mg_score -= 30 if pos.side else -30

    score = ((mg_score * (256 - game_phase_score)) + (eg_score * game_phase_score)) / 256
    # score = mg_score

    return -score if pos.side else score
