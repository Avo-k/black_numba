from constants import *
from moves import *
from bb_operations import get_ls1b_index
from evaluation import evaluate
from collections import namedtuple


def random_move(pos) -> int:
    """return a random legal move"""
    legal_moves = generate_legal_moves(pos)
    return np.random.choice(legal_moves) if legal_moves else None


@nb.experimental.jitclass([
    ("nodes", nb.uint64),
    ("ply", nb.uint64),
    ("killer_moves", nb.uint64[:, :]),
    ("history_moves", nb.uint64[:, :, :]),
    ("pv_table", nb.uint64[:, :]),
    ("pv_length", nb.uint8[:]),
    ("follow_pv", nb.b1),
    ("score_pv", nb.b1),
    ("hash_table", hash_numba_type[:])])
class Black_numba:
    def __init__(self):
        self.nodes = 0
        self.ply = 0
        # Killer moves [id][ply]
        self.killer_moves = np.zeros((2, MAX_PLY), dtype=np.uint64)
        # History moves [side][piece][square]
        self.history_moves = np.zeros((2, 6, 64), dtype=np.uint64)
        # Principal Variation (PV)
        self.pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint64)
        self.pv_length = np.zeros(MAX_PLY, dtype=np.uint8)
        self.follow_pv = False
        self.score_pv = False
        # Transposition Table
        self.hash_table = np.empty(MAX_HASH_SIZE, dtype=hash_numpy_type)

    def read_hash_entry(self, pos, depth, alpha, beta):
        entry = self.hash_table[pos.hash_key % MAX_HASH_SIZE]

        if entry.key == pos.hash_key:
            if entry.depth >= depth:
                if entry.flag == hash_flag_exact:
                    return entry.score
                if entry.flag == hash_flag_alpha and entry.score <= alpha:
                    return alpha
                if entry.flag == hash_flag_beta and entry.score >= beta:
                    return beta
        return no_hash_entry

    def write_hash_entry(self, pos, score, depth, hash_flag):

        i = pos.hash_key % MAX_HASH_SIZE

        self.hash_table[i].key = pos.hash_key
        self.hash_table[i].depth = depth
        self.hash_table[i].flag = hash_flag
        self.hash_table[i].score = score


@njit
def search(bot, pos, print_info=False, depth_max=15):
    """yield depth searched, best move, score (cp)"""
    bot.killer_moves = np.zeros((2, MAX_PLY), dtype=np.uint64)
    bot.history_moves = np.zeros((2, 6, 64), dtype=np.uint64)
    bot.pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint64)
    bot.pv_length = np.zeros(MAX_PLY, dtype=np.uint8)
    bot.score_pv = False
    bot.nodes = 0

    alpha, beta = -50000, 50000

    for depth in range(1, depth_max + 1):
        bot.follow_pv = True

        score = negamax(bot, pos, depth, alpha, beta)

        if score <= alpha or score >= beta:
            alpha, beta = -50000, 50000
            continue

        alpha, beta = score - 50, score + 50

        if print_info:
            pv_line = [get_move_uci(bot.pv_table[0][c]) for c in range(bot.pv_length[0])]
            print("info score cp", score, "depth", depth, "nodes", bot.nodes, "pv", " ".join(pv_line))

        # print("best move", get_move_uci(bot.pv_table[0][0]))
        yield depth, bot.pv_table[0][0], score


@njit
def quiescence(bot, pos, alpha, beta):
    bot.nodes += 1
    evaluation = evaluate(pos)

    if evaluation >= beta:
        return beta

    alpha = max(alpha, evaluation)

    move_list = [(m, score_move(bot, pos, m)) for m in generate_moves(pos)]
    move_list.sort(reverse=True, key=lambda m: m[1])

    for move, _ in move_list:
        new_pos = make_move(pos, move, only_captures=True)
        if new_pos is None:  # illegal move
            continue
        bot.ply += 1
        score = -quiescence(bot, new_pos, -beta, -alpha)
        bot.ply -= 1
        if score >= beta:
            return beta
        alpha = max(alpha, score)

    return alpha


@njit
def negamax(bot, pos, depth, alpha, beta):
    """return the value of a position given a certain depth
    using alpha-beta search and optimisations"""

    hash_flag = hash_flag_alpha
    hash_entry = bot.read_hash_entry(pos, depth, alpha, beta)

    if hash_entry != no_hash_entry:
        # This position has already been searched
        # at this depth or higher
        return hash_entry

    bot.pv_length[bot.ply] = bot.ply

    if depth == 0:
        return quiescence(bot, pos, alpha, beta)

    # We are way too deep for lots of arrays
    if bot.ply > MAX_PLY - 1:
        return evaluate(pos)

    bot.nodes += 1
    in_check = is_square_attacked(pos, get_ls1b_index(pos.pieces[pos.side][5]), pos.side ^ 1)
    if in_check:
        depth += 1

    legal_moves = 0

    # Null move pruning
    if depth >= 3 and not in_check and bot.ply:

        # try not moving
        null_pos = make_null_move(pos)
        score = -negamax(bot, null_pos, depth - reduction_limit, -beta, -beta + 1)

        if score >= beta:
            return beta

    move_list = generate_moves(pos)
    if bot.follow_pv:
        enable_pv_scoring(bot, move_list)

    # Decorate list with scores
    move_list = [(m, score_move(bot, pos, m)) for m in move_list]
    # Move ordering
    move_list.sort(reverse=True, key=lambda m: m[1])

    moves_searched = 0

    for move, _ in move_list:

        new_pos = make_move(pos, move)
        if new_pos is None:  # illegal move
            continue

        bot.ply += 1
        legal_moves += 1

        if moves_searched == 0:
            score = -negamax(bot, new_pos, depth - 1, -beta, -alpha)

        else:  # Late Move Reduction

            # check if the move is stable
            if moves_searched >= full_depth_moves and depth >= reduction_limit and\
                    not in_check and not get_move_capture(move) and not get_move_promote_to(move):
                # search with reduced depth and narrower window
                score = - negamax(bot, new_pos, depth - 2, -alpha - 1, -alpha)
            else:
                score = alpha + 1

            # Principal Variation Search (PVS)
            if score > alpha:   # if one of the late moves was actually good
                # research with narrower window
                score = -negamax(bot, new_pos, depth - 1, -alpha - 1, -alpha)

                if alpha < score < beta:    # the move was really good
                    # research with full depth
                    score = -negamax(bot, new_pos, depth - 1, -beta, -alpha)

        bot.ply -= 1
        moves_searched += 1

        # fail-hard beta cutoff
        if score >= beta:

            bot.write_hash_entry(pos, beta, depth, hash_flag_beta)

            # if quiet move
            if not get_move_capture(move):
                bot.killer_moves[1][bot.ply] = bot.killer_moves[0][bot.ply]
                bot.killer_moves[0][bot.ply] = move
            # fail high
            return beta

        if score > alpha:

            hash_flag = hash_flag_exact

            if not get_move_capture(move):
                # store history move
                bot.history_moves[pos.side][get_move_piece(move)][get_move_target(move)] += depth

            alpha = score
            bot.pv_table[bot.ply][bot.ply] = move

            for next_ply in range(bot.ply + 1, bot.pv_length[bot.ply + 1]):
                bot.pv_table[bot.ply][next_ply] = bot.pv_table[bot.ply + 1][next_ply]

            bot.pv_length[bot.ply] = bot.pv_length[bot.ply + 1]

    if legal_moves == 0:
        if in_check:  # checkmate
            return -50000 + bot.ply
        else:  # stalemate
            return 0

    bot.write_hash_entry(pos, alpha, depth, hash_flag)

    return alpha


@njit
def enable_pv_scoring(bot, move_list):
    bot.follow_pv = False
    for move in move_list:
        if bot.pv_table[0][bot.ply] == move:
            bot.score_pv = True
            bot.follow_pv = True


@njit(nb.uint64(Black_numba.class_type.instance_type, Position.class_type.instance_type, nb.uint64), cache=True)
def score_move(bot, pos, move) -> int:
    """
    return a score representing the move potential

    ----- Move ordering -----
    1. PV move
    2. Captures in MVV/LVA
    3. 1st and 2nd killer moves
    4. History moves
    5. Unsorted moves
    """

    if bot.score_pv:  # PV move
        if bot.pv_table[0][bot.ply] == move:
            bot.score_pv = False
            return 20000

    if get_move_capture(move):  # capture move
        attacker = get_move_piece(move)
        victim_square = get_move_target(move)
        victim = pawn  # in case of en-passant
        for p, bb in enumerate(pos.pieces[pos.side ^ 1]):
            if get_bit(bb, victim_square):
                victim = p
                break
        return mvv_lva[attacker][victim] + 10000

    else:  # quiet move
        if bot.killer_moves[0][bot.ply] == move:
            return 9000
        elif bot.killer_moves[1][bot.ply] == move:
            return 8000
        else:
            return bot.history_moves[get_move_side(move)][get_move_piece(move)][get_move_target(move)]


@njit
def print_move_scores(bot, pos):
    decorated_ml = [(m, score_move(bot, pos, m)) for m in generate_moves(pos)]
    decorated_ml.sort(reverse=True, key=lambda m: m[1])

    for move in decorated_ml:
        print("move:", get_move_uci(move[0]), "score:", move[1])
