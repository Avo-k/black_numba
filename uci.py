import sys
import argparse
import time
import math

from position import parse_fen, print_position
from constants import start_position
from moves import make_move, parse_move, get_move_uci
from search import Black_numba, random_move, search
from perft import uci_perft


def parse_position(command):
    """
    parse 'position' uci command
    eg: position startpos moves e2e4 e7e5 f1c4 g8f6
        position fen k7/6R1/2K5/8/8/8/8/8 w - - 16 9
    """

    param = command.split()
    index = command.find('moves')
    fen = ""

    move_list = [] if index == -1 else command[index:].split()[1:]

    if param[1] == "startpos":
        fen = start_position
    elif param[1] == "fen":
        fen_part = command if index == -1 else command[:index]
        _, _, fen = fen_part.split(maxsplit=2)

    pos = parse_fen(fen)

    for uci_move in move_list:
        move = parse_move(pos, uci_move)
        pos = make_move(pos, move)

    return pos


def parse_go(command, pos, bot):
    """parse 'go' uci command"""

    d = 12
    t = 1
    n = 10**9

    _, *params = command.split()

    # vivement 3.10!
    for p, v in zip(*2*(iter(params),)):
        print(p, v)
        if p == "perft":
            uci_perft(pos, depth=5)
            return
        elif p == "depth":
            d = int(v)
        elif p == "movetime":
            t = int(v)
        elif p == "nodes":
            n = int(v)
        elif p == "wtime":
            if not pos.side:
                t = int(v) // 40
        elif p == "btime":
            if pos.side:
                t = int(v) // 40

    _, move, _ = search(bot, pos, print_info=True, depth_limit=d, time_limit=t, node_limit=n)

    best_move = get_move_uci(move)
    ponder = get_move_uci(bot.pv_table[0][1])

    print(f"bestmove {best_move} ponder {ponder}")


def main():
    """
    The main input/output loop.
    This implements a slice of the UCI protocol.
    """

    pos = parse_fen(start_position)
    bot = Black_numba()

    while True:
        msg = input()
        print(f">>> {msg}", file=sys.stderr)

        if msg == "quit":
            break

        elif msg == "uci":
            print("id name black_numba")
            print("id name Avo-k")
            print("uciok")

        elif msg == "isready":
            print("readyok")

        elif msg == "ucinewgame":
            pos = parse_fen(start_position)
            bot = Black_numba()

        elif msg[:8] == "position":
            pos = parse_position(msg)
            bot = Black_numba()

        elif msg[:2] == "go":
            parse_go(msg, pos, bot)

        elif msg == "d":
            if pos is not None:
                print_position(pos)

        elif msg == "pv":
            print(bot.pv_table[0][:10])


if __name__ == "__main__":

    print("compiling...")
    compiling_time = time.perf_counter()
    search(Black_numba(), parse_fen(start_position), print_info=False, depth_limit=2)
    print(f"compiled in {time.perf_counter() - compiling_time:.2f} seconds")

    main()
