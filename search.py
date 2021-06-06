from constants import *
from move_gen import generate_moves, make_move
from evaluation import evaluate
from numba import njit
import math


@njit
def random_move(pos):
    """return a random legal move"""
    return np.random.choice(generate_moves(pos))


@njit
def negamax(pos, depth, alpha, beta):
    """return the best move given a position"""
    if not depth:
        return evaluate(pos)

    move_list = generate_moves(pos)
    if not move_list:
        return evaluate(pos)

    value = -10**5
    best_move = None

    for move in move_list:
        new_pos = make_move(pos, move)
        if new_pos is None:  # illegal move
            continue
        score = -negamax(pos, depth - 1, -beta, -alpha)
        if score > value:
            value = score
            best_move = move

        alpha = max(alpha, value)
        if alpha >= beta:
            break

    return best_move
