from constants import *
from move_gen import generate_moves, make_move
from evaluation import evaluate


def random_move(pos):
    """return a random legal move"""
    return np.random.choice(generate_moves(pos))


def negamax(pos, depth):
    if not depth:
        return evaluate(pos)

    move_list = generate_moves(pos)
    if not move_list:
        return evaluate(pos)

    value = float("-inf")
    best_move = None

    for move in move_list:
        new_pos = make_move(pos, move)
        if not new_pos:  # illegal move
            continue

        score = negamax(pos, depth - 1)

        if score > value:
            value = score
            best_move = move

    return best_move
