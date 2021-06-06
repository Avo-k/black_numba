import time
import chess

from position import parse_fen
from move_gen import *


def perft(board, depth, b=None, print_info=False):
    if depth == 0:
        return 1
    count = 0
    moves = generate_moves(board)
    for m in moves:
        new_board = make_move(board, m)
        if new_board:
            if b: b.push_uci(get_move_uci(m))
            c = child_perft(new_board, depth - 1, b)
            if b: b.pop()
            count += c
            if print_info:
                print(f"move: {get_move_uci(m)}     nodes: {c}")
    return count


def child_perft(board, depth, b=None):
    if depth == 0:
        return 1
    count = 0
    moves = generate_moves(board)
    if b:
        PC_moves = set([m.uci() for m in b.legal_moves])
        BN_moves = set([get_move_uci(m) for m in generate_legal_moves(board)])
        if PC_moves != BN_moves:
            print([m.uci() for m in b.move_stack])
            if PC_moves - BN_moves:
                print("BN does not see:", PC_moves - BN_moves)
            else:
                print("BN thinks she can play:", BN_moves - PC_moves)
            raise Exception

    for m in moves:
        new_board = make_move(board, m)
        if new_board:
            if b:
                b.push_uci(get_move_uci(m))
            count += child_perft(new_board, depth - 1, b)
            if b:
                b.pop()
    return count


positions = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":
        {1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609, 6: 119060324},
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1":
        {1: 48, 2: 2039, 3: 97862, 4: 4085603, 5: 193690690, 6: 8031647685},
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1":
        {1: 14, 2: 191, 3: 2812, 4: 43238, 5: 674624, 6: 11030083},
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1":
        {1: 6, 2: 264, 3: 9467, 4: 422333, 5: 15833292, 6: 706045033},
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8":
        {1: 44, 2: 1486, 3: 62379, 4: 2103487, 5: 89941194}
}

# "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10":
# {1: 46, 2: 2079, 3: 89890, 4: 3894594, 5: 164075551}


def main():
    for i, (pos, t) in enumerate(positions.items(), 1):
        print("-" * 30)
        print(" " * 8, f"POSITION {i}")
        print("-" * 30)

        for depth, result in t.items():
            position = parse_fen(pos)
            if depth > 3:
                continue
            s = time.time()
            b = chess.Board(fen=pos)
            r = perft(position, depth, b=None)
            if depth > 2:
                print("depth       time       n/s")
                print(f"  {depth}        {round(time.time() - s, 3)}       {round(result / (time.time() - s))}")

            assert r == result


if __name__ == "__main__":
    main()
