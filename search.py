import time

from constants import *
import constants
from moves import *
from bb_operations import get_ls1b_index
from evaluation import evaluate, get_game_phase_score


def random_move(pos) -> int:
    """return a random legal move"""
    legal_moves = generate_legal_moves(pos)
    return np.random.choice(legal_moves) if legal_moves else None


@nb.experimental.jitclass([
    ("nodes", nb.uint64),
    ("ply", nb.uint32),
    ("killer_moves", nb.uint64[:, :]),
    ("history_moves", nb.uint8[:, :, :]),
    ("pv_table", nb.uint64[:, :]),
    ("pv_length", nb.uint64[:]),
    ("follow_pv", nb.b1),
    ("score_pv", nb.b1),
    ("hash_table", hash_numba_type[:]),
    ("repetition_table", nb.uint64[:]),
    ("repetition_index", nb.uint16),
    ("time_limit", nb.uint64),
    ("node_limit", nb.uint64),
    ("start", nb.uint64),
    ("stopped", nb.b1)])
class Black_numba:
    def __init__(self):
        self.nodes = 0
        self.ply = 0
        # Killer moves [id][ply]
        self.killer_moves = np.zeros((2, MAX_PLY), dtype=np.uint64)
        # History moves [side][piece][square]
        self.history_moves = np.zeros((2, 6, 64), dtype=np.uint8)
        # Principal Variation (PV)
        self.pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint64)
        self.pv_length = np.zeros(MAX_PLY, dtype=np.uint64)
        self.follow_pv = False
        self.score_pv = False
        # Transposition Table
        self.hash_table = np.zeros(MAX_HASH_SIZE, dtype=hash_numpy_type)
        # Repetitions
        self.repetition_table = np.zeros(1000, dtype=np.uint64)
        self.repetition_index = 0
        # Time management
        self.time_limit = 1000
        self.node_limit = 10**7
        self.start = 0
        self.stopped = True

    def reset_bot(self, time_limit, node_limit):
        self.killer_moves = np.zeros((2, MAX_PLY), dtype=np.uint64)
        self.history_moves = np.zeros((2, 6, 64), dtype=np.uint8)
        self.pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint64)
        self.pv_length = np.zeros(MAX_PLY, dtype=np.uint64)
        self.score_pv = False
        self.nodes = 0
        self.stopped = False
        self.time_limit = time_limit
        self.node_limit = node_limit
        with nb.objmode(start=nb.uint64):
            start = time.time() * 1000
        self.start = start

    def read_hash_entry(self, pos, depth, alpha, beta):
        entry = self.hash_table[pos.hash_key % MAX_HASH_SIZE]

        if entry.key == pos.hash_key:
            if entry.depth >= depth:

                score = entry.score
                if score < -LOWER_MATE:
                    score += self.ply
                elif score > LOWER_MATE:
                    score -= self.ply

                if entry.flag == hash_flag_exact:
                    return score
                if entry.flag == hash_flag_alpha and entry.score <= alpha:
                    return alpha
                if entry.flag == hash_flag_beta and entry.score >= beta:
                    return beta
        return no_hash_entry

    def write_hash_entry(self, pos, score, depth, hash_flag):

        i = pos.hash_key % MAX_HASH_SIZE

        if score < -LOWER_MATE:
            score -= self.ply
        elif score > LOWER_MATE:
            score += self.ply

        self.hash_table[i].key = pos.hash_key
        self.hash_table[i].depth = depth
        self.hash_table[i].flag = hash_flag
        self.hash_table[i].score = score

    def is_repetition(self, pos):
        if pos.hash_key in self.repetition_table[:self.repetition_index]:
            return True
        return False

    def communicate(self):
        with nb.objmode(spent=nb.uint64):
            spent = time.perf_counter() * 1000 - self.start
        if spent > self.time_limit or self.nodes > self.node_limit:
            self.stopped = True


@njit
def enable_pv_scoring(bot, move_list):
    bot.follow_pv = False

    if bot.pv_table[0][bot.ply] in move_list:
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


@njit
def quiescence(bot, pos, alpha, beta):

    if not bot.nodes & time_precision:
        bot.communicate()

    bot.nodes += 1

    # We are way too deep for lots of arrays
    if bot.ply > MAX_PLY - 1:
        return evaluate(pos)

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

        bot.repetition_index += 1
        bot.repetition_table[bot.repetition_index] = pos.hash_key

        score = -quiescence(bot, new_pos, -beta, -alpha)

        bot.ply -= 1
        bot.repetition_index -= 1

        if bot.stopped:
            return 0

        if score > alpha:
            alpha = score
            if score >= beta:
                return beta

    return alpha


@njit
def negamax(bot, pos, depth, alpha, beta):
    """return the value of a position given a certain depth
    using alpha-beta search and optimisations"""

    hash_flag = hash_flag_alpha

    if bot.ply and bot.is_repetition(pos):
        return 0

    hash_entry = bot.read_hash_entry(pos, depth, alpha, beta)

    pv_node = beta - alpha > 1

    if bot.ply and hash_entry != no_hash_entry and not pv_node:
        # This position has already been searched
        # at this depth or higher
        return hash_entry

    if not bot.nodes & time_precision:
        bot.communicate()

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

        bot.ply += 1

        bot.repetition_index += 1
        bot.repetition_table[bot.repetition_index] = pos.hash_key

        score = -negamax(bot, null_pos, depth - 1 - 2, -beta, -beta + 1)
        bot.ply -= 1
        bot.repetition_index -= 1

        if bot.stopped:
            return 0

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
        bot.repetition_index += 1
        bot.repetition_table[bot.repetition_index] = pos.hash_key
        legal_moves += 1

        if moves_searched == 0:
            score = -negamax(bot, new_pos, depth - 1, -beta, -alpha)

        else:  # Late Move Reduction

            # condition to consider LMR
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
        bot.repetition_index -= 1

        if bot.stopped:
            return 0

        moves_searched += 1

        if score > alpha:

            hash_flag = hash_flag_exact

            if not get_move_capture(move):
                # store history move
                bot.history_moves[pos.side][get_move_piece(move)][get_move_target(move)] += depth

            alpha = score
            # write PV node
            bot.pv_table[bot.ply][bot.ply] = move

            for next_ply in range(bot.ply + 1, bot.pv_length[bot.ply + 1]):
                bot.pv_table[bot.ply][next_ply] = bot.pv_table[bot.ply + 1][next_ply]

            bot.pv_length[bot.ply] = bot.pv_length[bot.ply + 1]

            # fail-hard beta cutoff
            if score >= beta:

                bot.write_hash_entry(pos, beta, depth, hash_flag_beta)

                # if quiet move
                if not get_move_capture(move):
                    bot.killer_moves[1][bot.ply] = bot.killer_moves[0][bot.ply]
                    bot.killer_moves[0][bot.ply] = move
                # fail high
                return beta

    if legal_moves == 0:
        if in_check:  # checkmate
            return -UPPER_MATE + bot.ply
        else:  # stalemate
            return 0

    bot.write_hash_entry(pos, alpha, depth, hash_flag)

    return alpha


@njit
def search(bot, pos, print_info=False, depth_limit=32, time_limit=1000, node_limit=10**7):
    """yield depth searched, best move, score (cp)"""

    bot.reset_bot(time_limit=time_limit, node_limit=node_limit)

    depth, value = 0, 0
    alpha, beta = -BOUND_INF, BOUND_INF

    for depth in range(1, depth_limit + 1):
        if bot.stopped or not -LOWER_MATE < value < LOWER_MATE:
            break
        bot.follow_pv = True

        value = negamax(bot, pos, depth, alpha, beta)

        if value <= alpha or value >= beta:
            alpha, beta = -BOUND_INF, BOUND_INF
            continue
        alpha, beta = value - 50, value + 50

        if print_info:
            pv_line = " ".join([get_move_uci(bot.pv_table[0][c]) for c in range(bot.pv_length[0])])
            s_score = "mate"
            if -UPPER_MATE < value < -LOWER_MATE:
                score = -(value + UPPER_MATE) // 2
            elif LOWER_MATE < value < UPPER_MATE:
                score = (UPPER_MATE - value) // 2 + 1
            else:
                s_score = "cp"
                score = value

            print("info", "depth", depth, "score", s_score, int(score), "nodes", bot.nodes, "pv", pv_line)

            # with nb.objmode(ms_spent=nb.float64):
            #     ms_spent = time.time() * 1000 - bot.start
            # nps = int(bot.nodes / ms_spent * 1000)
            # print("info", "depth", depth, "score", s_score, score, "nodes", bot.nodes,
            #       "nps", nps, "time", int(ms_spent), "pv", pv_line)

    # print(score == bot.read_hash_entry(pos, depth, alpha, beta))
    return depth, bot.pv_table[0][0], score
