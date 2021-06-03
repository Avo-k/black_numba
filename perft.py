import time
from tqdm import tqdm

from position import parse_fen, print_position
from move_gen import *

captures = 0
ep = 0
castles = 0
promo = 0
pushpush = 0
checks = 0
checkmates = 0


def perft(board, depth, debug=False):
    if depth == 0:
        return 1
    count = 0
    moves = generate_moves(board)
    for m in tqdm(moves):
        new_board = make_move(board, m)
        if new_board:

            if get_move_target(m) == d3:
                c = perft(new_board, depth - 1, True)
                print_move(m)
                count += c
                print(c)

            elif debug:
                c = child_perft(new_board, depth - 1)
                print_move(m)
                count += c
                print(c)

            else:
                c = child_perft(new_board, depth - 1)
                print_move(m)
                count += c
                # print(c)

    return count


def child_perft(board, depth):
    global captures, ep, castles, promo, checks, pushpush
    if depth == 0:
        return 1
    count = 0
    moves = generate_moves(board)
    for m in moves:
        new_board = make_move(board, m)
        if new_board:
            count += child_perft(new_board, depth - 1)

            # if get_move_promote_to(m):
            #     promo += 1
            # if get_move_capture(m):
            #     captures += 1
            # if get_move_double(m):
            #     pushpush += 1
            # if get_move_enpas(m):
            #     ep += 1
            # if get_move_castling(m):
            #     castles += 1
            # k = get_ls1b_index(new_board.pieces[new_board.side][king])
            # if is_square_attacked(new_board, k, white if new_board.side else black):
            #     checks += 1

    return count


pos1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
time1 = {1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609}
pos2 = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
time2 = {1: 48, 2: 2039, 3: 97862, 4: 4085603, 5: 193690690}
pos3 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
time3 = {1: 14, 2: 191, 3: 2812, 4: 43238, 5: 674624}

positions = [(pos1, time1), (pos2, time2), (pos3, time3)]


def main():
    for i, (pos, t) in enumerate(positions, 1):
        if i > 1:
            break
        print("-" * 30)
        print(" " * 8, f"POSITION {i}")
        print("-" * 30)

        for depth, result in t.items():
            position = parse_fen(pos)
            if depth == 4:
                break
            s = time.time()
            r = perft(position, depth)
            print(r, result)
            print(f"time depth {depth} {round(time.time() - s, 3)} sec")
            assert r == result
            # print(f"captures    ep    castles    promo    checks    pushpush")
            # print(f"{captures}          {ep}         {castles}         {promo}        {checks}       {pushpush}")


if __name__ == "__main__":
    main()
