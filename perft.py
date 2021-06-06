import time
import chess
import sys

from constants import *
from position import parse_fen
from move_gen import generate_moves, make_move, generate_legal_moves, get_move_uci


positions = nb.typed.Dict.empty(key_type=nb.types.string, value_type=nb.types.uint64[:])

positions["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] = \
    np.array([1, 20, 400, 8902, 197281, 4865609, 119060324], dtype=np.uint64)
positions["r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"] = \
    np.array([1, 48, 2039, 97862, 4085603, 193690690, 8031647685], dtype=np.uint64)
positions["8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"] = \
    np.array([1, 14, 191, 2812, 43238, 674624, 11030083], dtype=np.uint64)
positions["r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"] = \
    np.array([1, 6, 264, 9467, 422333, 15833292, 706045033], dtype=np.uint64)
positions["rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"] = \
    np.array([1, 44, 1486, 62379, 2103487, 89941194], dtype=np.uint64)
positions["r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"] = \
    np.array([1, 46, 2079, 89890, 3894594, 164075551], dtype=np.uint64)


def debug_perft(board, depth, b, print_info=False):
    """perft test with python-chess in parallel to narrow down the bugs"""
    if depth == 0:
        return 1
    count = 0
    moves = generate_legal_moves(board)
    PC_moves = set(m.uci() for m in b.legal_moves)
    BN_moves = set(get_move_uci(m) for m in moves)
    if PC_moves != BN_moves:
        print("Moves played:", [m.uci() for m in b.move_stack])
        print("In this position:", b.fen())
        if PC_moves - BN_moves:
            print("BN does not see:", PC_moves - BN_moves)
        else:
            print("BN thinks she can play:", BN_moves - PC_moves)
        sys.exit()

    for m in moves:
        b.push_uci(get_move_uci(m))
        c = debug_perft(make_move(board, m), depth - 1, b)
        count += c
        b.pop()
        if print_info:
            print(f"move: {get_move_uci(m)}     nodes: {c}")
    return count


@nb.njit
def compiled_perft(board, depth):
    """fast compiled perft test"""
    if depth == 0:
        return 1
    count = 0
    moves = generate_moves(board)
    for m in moves:
        new_board = make_move(board, m)
        if new_board is not None:
            c = compiled_perft(new_board, depth - 1)
            count += c
    return count


def debug_iterative_perft(depth_max=3):
    for i, (pos, t) in enumerate(positions.items(), 1):
        # if i in (2, 5, 6):
        #     continue
        print("-" * 30)
        print(" " * 8, f"POSITION {i}")
        print("-" * 30)

        for depth, result in enumerate(t):
            position = parse_fen(pos)
            if depth > depth_max:
                continue
            s = time.time()
            b = chess.Board(fen=pos)
            r = debug_perft(position, depth, b, print_info=False)
            if depth > 2:
                print("depth     time(s)       n/s")
                print(f"  {depth}        {round(time.time() - s, 3)}       {round(result / (time.time() - s))}")
            assert r == result


def fast_iterative_perft(depth_max=4):
    for i, (pos, t) in enumerate(positions.items(), 1):
        if i == 6:
            continue
        print("-" * 30)
        print(" " * 8, "POSITION", i)
        print("-" * 30)

        for depth, result in enumerate(t):
            position = parse_fen(pos)
            if depth > depth_max:
                continue
            s = time.time()
            r = compiled_perft(position, depth)
            if depth > 2:
                print("depth     time(s)       n/s")
                print(f"  {depth}        {round(time.time() - s, 3)}      {round(result / (time.time() - s))}")
            assert r == result


if __name__ == "__main__":
    debug_iterative_perft()
    # fast_iterative_perft()
