from constants import *
from move_gen import *
from bb_operations import get_ls1b_index
from evaluation import evaluate
from collections import namedtuple


def random_move(pos) -> int:
    """return a random legal move"""
    legal_moves = generate_legal_moves(pos)
    return np.random.choice(legal_moves) if legal_moves else None


Entry = namedtuple('Entry', 'lower upper')

Bot = namedtuple('bot', 'best_move nodes ply')


@nb.experimental.jitclass([
    ("nodes", nb.uint64),
    ("ply", nb.uint64),
    ("killer_moves", nb.uint64[:, :]),
    ("history_moves", nb.uint64[:, :, :]),
    ("pv_table", nb.uint64[:, :]),
    ("pv_lenght", nb.uint8[:])])
class Black_numba:
    def __init__(self):
        self.nodes = 0
        self.ply = 0
        # killer moves [id][ply]
        self.killer_moves = np.zeros((2, MAX_PLY), dtype=np.uint64)
        # history moves [side][piece][square]
        self.history_moves = np.zeros((2, 6, 64), dtype=np.uint64)
        # Principal Variation (PV)
        self.pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint64)
        self.pv_lenght = np.zeros(MAX_PLY, dtype=np.uint8)
        # self.follow_pv = 0
        # self.score_pv = 0


@njit
def search(bot, pos, print_info=False):
    """yield depth searched, best move, score (cp)"""
    bot.killer_moves = np.zeros((2, MAX_PLY), dtype=np.uint64)
    bot.history_moves = np.zeros((2, 6, 64), dtype=np.uint64)
    bot.pv_table = np.zeros((MAX_PLY, MAX_PLY), dtype=np.uint64)
    bot.pv_lenght = np.zeros(MAX_PLY, dtype=np.uint8)
    # bot.follow_pv = 0
    # bot.score_pv = 0
    bot.nodes = 0

    for depth in range(1, 100):

        score = negamax(bot, pos, depth, -10 ** 6, 10 ** 6)

        if print_info:
            pv_line = [get_move_uci(bot.pv_table[0][c]) for c in range(depth)]
            print("info score cp", score, "depth", depth, "nodes", bot.nodes, "pv", " ".join(pv_line))

        # print("best move", get_move_uci(bot.pv_table[0][0]))
        yield depth, bot.pv_table[0][0], score


@njit
def quiescence(bot, pos, alpha, beta):
    evaluation = evaluate(pos)
    bot.nodes += 1

    if evaluation >= beta:
        return beta

    alpha = max(alpha, evaluation)



    move_list = [(m, score_move(bot, pos, m)) for m in generate_moves(pos)]
    move_list.sort(reverse=True, key=lambda m: m[1])

    for move in move_list:
        new_pos = make_move(pos, move[0], only_captures=True)
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
    """return the best move given a position"""
    if depth == 0:
        return quiescence(bot, pos, alpha, beta)

    # We are way too deep
    if bot.ply > MAX_PLY - 1:
        return evaluate(pos)

    bot.nodes += 1
    bot.pv_lenght[bot.ply] = bot.ply
    in_check = is_square_attacked(pos, get_ls1b_index(pos.pieces[pos.side][5]), pos.side ^ 1)
    legal_moves = 0
    # alphaorig = alpha

    if in_check: depth += 1

    move_list = [(m, score_move(bot, pos, m)) for m in generate_moves(pos)]
    move_list.sort(reverse=True, key=lambda m: m[1])

    for move in move_list:
        move = move[0]
        bot.ply += 1
        new_pos = make_move(pos, move)
        if new_pos is None:  # illegal move
            bot.ply -= 1
            continue
        legal_moves += 1
        score = -negamax(bot, new_pos, depth - 1, -beta, -alpha)
        bot.ply -= 1

        # fail-hard beta cutoff
        if score >= beta:
            if not get_move_capture(move):
                bot.killer_moves[1][bot.ply] = bot.killer_moves[0][bot.ply]
                bot.killer_moves[0][bot.ply] = move
            return beta

        if score > alpha:
            if not get_move_capture(move):
                bot.history_moves[pos.side][get_move_piece(move)][get_move_target(move)] += depth
            alpha = score

            bot.pv_table[bot.ply][bot.ply] = move
            for next_ply in range(bot.ply + 1, bot.pv_lenght[bot.ply + 1]):
                bot.pv_table[bot.ply][next_ply] = bot.pv_table[bot.ply + 1][next_ply]
            bot.pv_lenght[bot.ply] = bot.pv_lenght[bot.ply + 1]

    if legal_moves == 0:
        if in_check:  # checkmate
            return -10 ** 6 + bot.ply
        else:  # stalemate
            return 0

    return alpha


def enable_pv_scoring(bot, move_list):
    bot.follow_pv = 0

    for move in move_list:
        if bot.pv_table[0][bot.ply] == move:
            score_pv = 1


@njit(nb.uint64(Black_numba.class_type.instance_type, Position.class_type.instance_type, nb.uint64), cache=True)
def score_move(bot, pos, move) -> int:
    """return a score representing the move potential"""

    if get_move_capture(move):  # capture move
        attacker = get_move_piece(move)
        victim_square = get_move_target(move)
        victim = pawn  # in case of en-passant
        for p, bb in enumerate(pos.pieces[pos.side ^ 1]):
            if get_bit(bb, victim_square):
                victim = p
                break
        return mvv_lva[attacker][victim] + 10 ** 4

    else:  # quiet move
        if bot.killer_moves[0][bot.ply] == move:
            return 9000
        elif bot.killer_moves[1][bot.ply] == move:
            return 8000
        else:
            return bot.history_moves[get_move_side(move)][get_move_piece(move)][get_move_target(move)]


def print_move_scores(bot, pos):
    move_list = generate_moves(pos)

    decorated_ml = [(m, score_move(bot, pos, m)) for m in move_list]

    decorated_ml.sort(reverse=True, key=lambda m: m[1])

    for move in decorated_ml:
        print(f"move: {get_move_uci(move[0])}  score: {move[1]}")


# to delete
@njit
def sorted_move_list(bot, pos, move_list):
    """return a sorted move list using the Decorate-Sort-Undecorate method"""

    decorated_ml = [(m, score_move(bot, pos, m)) for m in move_list]

    decorated_ml.sort(reverse=True, key=lambda m: m[1])

    return [m[0] for m in decorated_ml]
