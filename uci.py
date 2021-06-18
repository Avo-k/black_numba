from position import parse_fen
from constants import start_position
from moves import generate_moves, get_move_source, get_move_target, get_move_promote_to, make_move
from search import random_move


def main():

    print("id name black_numba")
    print("id name Avo-k")
    print("uciok")

    while True:
        break


def parse_move(pos, uci_move: str) -> int:
    """encode a uci move"""

    source = (ord(uci_move[0]) - ord('a')) + ((8 - int(uci_move[1])) * 8)
    target = (ord(uci_move[2]) - ord('a')) + ((8 - int(uci_move[3])) * 8)

    for move in generate_moves(pos):
        if get_move_source(move) == source and get_move_target(move) == target:
            promoted_piece = get_move_promote_to(move)
            if promoted_piece:
                for p, s in enumerate(('n', 'b', 'r', 'q'), 1):
                    if promoted_piece == p and uci_move[4] == s:
                        return move
                return 0    # in case of illegal promotion (e.g. e7d8f)
            return move
    return 0


def parse_position(command):
    """parse 'position' uci command"""

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

    for smove in move_list:
        move = parse_move(pos, smove)
        pos = make_move(pos, move)

    return pos


def parse_go(command, pos):
    """parse 'go' uci command"""

    depth = 4
    _, *params = command.split()

    for p, v in zip(*2*(iter(params),)):
        if p == "depth":
            depth = int(v)

    print("bestmove", random_move(pos))


if __name__ == "__main__":
    main()

