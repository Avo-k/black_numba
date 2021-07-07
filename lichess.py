import berserk
import time
import sys
import chess.polyglot
import threading

from position import parse_fen, print_position
from constants import start_position
from search import Black_numba, search, random_move
from moves import get_move_uci, make_move
from uci import parse_move

API_TOKEN = open("api_token.txt").read()
bot_id = 'black_numba'

session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)


class Game:
    def __init__(self, client, game_id):
        self.game_id = game_id
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        self.current_state = next(self.stream)
        self.bot_is_white = self.current_state['white'].get('id') == bot_id
        self.time_str = "wtime" if self.bot_is_white else "btime"
        self.moves = ""
        self.bot = Black_numba()
        self.pos = parse_fen(start_position)
        self.theory = True

    def run(self):
        # From Position
        if self.current_state['variant']['short'] == "FEN":
            self.pos = parse_fen(self.current_state['initialFen'])
            if not self.pos.side:
                self.make_first_move()

        # Reconnect to a game
        elif self.moves != self.current_state['state']['moves']:
            move_list = self.current_state['state']['moves'].split()
            for smove in move_list:
                move = parse_move(self.pos, smove)
                self.pos = make_move(self.pos, move)
            self.make_first_move()

        # if you have to start
        elif self.bot_is_white != self.pos.side:
            self.make_first_move()

        # ponder_thread = threading.Thread(target=self.ponder)

        # main game loop
        for event in self.stream:
            if event['type'] == 'gameState':
                if event['status'] == 'started':
                    if not event['moves'] == self.moves:
                        new_move = parse_move(self.pos, event['moves'][len(self.moves):].strip())
                        self.moves = event['moves']
                        self.pos = make_move(self.pos, new_move)
                        bot_turn = self.bot_is_white != self.pos.side

                        if bot_turn:
                            self.play(event)

                            # print("my turn")
                            # if ponder_thread.is_alive():
                            #     assert threading.active_count() == 2
                            #     print("i am killing it")
                            #     self.bot.stopped = True
                            #     s = time.perf_counter_ns()
                            #     ponder_thread.join()
                            #     print(f"killed in {(time.perf_counter_ns() - s) / 10**6:.0f} ms")
                            #     # del ponder_thread
                            #     assert threading.active_count() == 1
                            # self.play(event)

                        # else:
                        #     ponder_thread = threading.Thread(target=self.ponder)
                        #     print("opp turn")
                        #     ponder_thread.start()

                elif event['status'] in ('mate', 'resign', 'outoftime', 'aborted', 'draw'):
                    print(event['status'])
                    break
                else:
                    print('NEW', event['status'])
                    break

    def play_random_fast(self):
        move = random_move(self.pos)
        client.bots.make_move(self.game_id, get_move_uci(move))

    def ponder(self, game_state):
        print("i am pondering")

        # set time limit
        remaining_time = game_state[self.time_str].timestamp()
        time_limit = remaining_time / 40 * 1000

        start = time.perf_counter_ns()
        depth, move, score = search(self.bot, self.pos, print_info=False, time_limit=time_limit)
        time_spent_ms = (time.perf_counter_ns() - start) / 10 ** 6
        print(f"pondering time:  {time_spent_ms:.0f}")
        print(f"pondering depth: {depth} - kns: {self.bot.nodes / time_spent_ms:.0f}")
        print("-" * 40)

    def play(self, game_state):
        # Look in the books
        if self.theory:
            entry = self.look_in_da_book(self.moves.split())
            if entry:
                self.client.bots.make_move(game_id, entry.move)
                print("still theory")
                return
            self.theory = False
            print("end of theory")

        # set time limit
        remaining_time = game_state[self.time_str].timestamp()
        time_limit = remaining_time / 40 * 1000

        # look for a move
        start = time.perf_counter_ns()
        depth, move, score = search(self.bot, self.pos, print_info=True, time_limit=time_limit)
        time_spent_ms = (time.perf_counter_ns() - start) / 10**6

        try:
            client.bots.make_move(self.game_id, get_move_uci(move))
        except berserk.exceptions.ResponseError as e:  # you flagged
            print(e)
            return

        print(f"time: {time_spent_ms:.0f} - kns: {self.bot.nodes / time_spent_ms:.0f}")
        # print(f"time delta: {time_spent_ms - time_limit:.0f} ms")
        print("-" * 40)

    def make_first_move(self):
        # Look in the books
        if self.theory:
            entry = self.look_in_da_book(self.moves.split())
            if entry:
                self.client.bots.make_move(game_id, entry.move)
                print("still theory")
                return
            self.theory = False
            print("end of theory")

        # look for a move
        print("playing 1st move")
        depth, move, score = search(self.bot, self.pos, print_info=True)

        client.bots.make_move(self.game_id, get_move_uci(move))

    @staticmethod
    def look_in_da_book(moves):
        fruit = chess.polyglot.open_reader("book/book_small.bin")
        board = chess.Board()
        for m in moves:
            board.push_uci(m)
        return fruit.get(board)


print("id name black_numba")
print("id author Avo-k")
print("compiling...")
compiling_time = time.time()
search(Black_numba(), parse_fen(start_position), print_info=False, depth_limit=1)
print(f"compiled in {time.time() - compiling_time:.2f} seconds")

for event in client.bots.stream_incoming_events():
    if event['type'] == 'challenge':
        challenge = event['challenge']
        if challenge['speed'] in ('bullet', 'blitz', 'rapid', 'classic'):
            if challenge['variant']['short'] in ("Std", "FEN"):
                client.bots.accept_challenge(challenge['id'])
                print('challenge accepted!')
        else:
            client.bots.decline_challenge(challenge['id'])

    elif event['type'] == 'gameStart':
        print(event['type'])
        game_id = event['game']['id']
        game = Game(client=client, game_id=game_id)
        game.run()
        del game

    else:  # challengeDeclined, gameFinish, challengeCanceled
        print(event['type'])
