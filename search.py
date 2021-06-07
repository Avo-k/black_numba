from constants import *
from move_gen import generate_moves, generate_legal_moves, make_move
from evaluation import evaluate


def random_move(pos) -> int:
    """return a random legal move"""

    legal_moves = generate_legal_moves(pos)

    return np.random.choice(legal_moves) if legal_moves else None


# @nb.experimental.jitclass([("best_move", nb.uint64)])
class Black_numba:
    def __init__(self):
        self.best_move = 0

    # @nb.jit
    def negamax(self, pos, depth, alpha, beta, ply=0):
        """return the best move given a position"""
        if depth == 0:
            return evaluate(pos)

        alphaorig = alpha
        best_so_far = 0

        move_list = generate_moves(pos)

        for move in move_list:
            new_pos = make_move(pos, move)
            if new_pos is None:  # illegal move
                continue
            ply += 1
            score = -self.negamax(new_pos, depth - 1, -beta, -alpha, ply)
            ply -= 1
            if score >= beta:
                return beta

            if score > alpha:
                alpha = score

                if ply == 0:
                    best_so_far = move

        if alphaorig != alpha:
            self.best_move = best_so_far

        return alpha

    def search(self, pos, depth):
        score = self.negamax(pos, depth, -10**6, 10**6)
        return self.best_move
