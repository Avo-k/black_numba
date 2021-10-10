import berserk
import time
import chess.polyglot
import requests
import sys
import threading

from position import parse_fen, print_position
from constants import start_position, stopped
from search import Black_numba, search, random_move
from moves import get_move_uci, make_move, parse_move
from bb_operations import count_bits

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
        self.book_moves = 0
        self.pcb = chess.Board()

    def run(self):
        # From Position
        if self.current_state['variant']['short'] == "FEN":
            fen = self.current_state['initialFen']
            print("from pos: ", fen)
            self.pos = parse_fen(fen)
            self.pcb.set_fen(fen)
            self.theory = False
            if self.bot_is_white != self.pos.side:
                self.play(remaining_time=self.current_state['state'][self.time_str] // 1000)

        # Reconnect to a game
        elif self.moves != self.current_state['state']['moves']:
            self.moves = self.current_state['state']['moves']
            move_list = self.moves.split()
            for smove in move_list:
                self.pcb.push_uci(smove)
                move = parse_move(self.pos, smove)
                self.pos = make_move(self.pos, move)
            print(self.current_state)
            self.play(remaining_time=self.current_state['state'][self.time_str] // 1000)

        # if you have to start
        elif self.bot_is_white != self.pos.side:
            self.play(remaining_time=self.current_state['state'][self.time_str] // 1000)

        # ponder_thread = threading.Thread(target=self.ponder)

        # main game loop
        for event in self.stream:
            if event['type'] == 'gameState':
                if event['status'] == 'started':
                    if not event['moves'] == self.moves:
                        s_move = event['moves'][len(self.moves):].strip()
                        self.moves = event['moves']
                        self.pos = make_move(self.pos, parse_move(self.pos, s_move))
                        self.pcb.push_uci(s_move)
                        remaining_time = event[self.time_str].timestamp()
                        bot_turn = self.bot_is_white != self.pos.side

                        if bot_turn:
                            # print("my turn")
                            # if ponder_thread.is_alive():
                            #     assert threading.active_count() == 2
                            #     print("i am killing it")
                            #     self.bot.stopped = True
                            #     constants.stopped = True
                            #     s = time.perf_counter_ns()
                            #     ponder_thread.join()
                            #     print(f"killed in {(time.perf_counter_ns() - s) / 10**6:.0f} ms")
                            #     constants.stopped = False
                            #     assert threading.active_count() == 1
                            self.play(remaining_time=remaining_time)

                        else:
                            if self.theory:
                                self.ponder(remaining_time=remaining_time)
                        #     self.ponder(remaining_time=event[self.time_str].timestamp())
                        #     print("opp turn")
                            # ponder_thread = threading.Thread(target=self.ponder)
                            # ponder_thread.start()

                elif event['status'] in ('mate', 'resign', 'outoftime', 'aborted', 'draw', 'stalemate'):
                    print(event['status'])
                    break
                else:
                    print('NEW', event['status'])
                    break

    def play_random_fast(self):
        move = random_move(self.pos)
        client.bots.make_move(self.game_id, get_move_uci(move))

    def ponder(self, remaining_time):
        # set time limit
        time_limit = 800
        start = time.perf_counter_ns()
        search(self.bot, self.pos, print_info=False, time_limit=time_limit, depth_limit=8)
        time_spent_ms = (time.perf_counter_ns() - start) / 10 ** 6
        print(f"pondering time:  {time_spent_ms:.0f}")

    def play(self, remaining_time):
        move_list = self.moves.split()
        start = time.perf_counter_ns()

        # Look in the books
        if self.theory:
            entry = self.look_in_da_book()
            if entry:
                move = entry.move
                time_spent_ms = (time.perf_counter_ns() - start) / 10 ** 6
                print(f"info book weight {entry.weight}")
                self.book_moves += 2
            else:
                self.theory = False
                print("end of theory")
                self.play(remaining_time)
                return

        # End-game table
        elif count_bits(self.pos.occupancy[2]) < 8:
            entry = self.syzygy()
            move = entry['uci']
            time_spent_ms = (time.perf_counter_ns() - start) / 10**6
            print(f"info syzygy wdl {entry['wdl']} dtm {entry['dtm']}")

        else:
            # time-management
            n_moves = min(10, len(move_list) - self.book_moves)
            factor = 2 - n_moves / 10
            target = remaining_time / 40 * 1000
            time_limit = round(factor * target)

            # look for a move
            depth, move, score = search(self.bot, self.pos, print_info=True, time_limit=time_limit)
            move = get_move_uci(move)
            time_spent_ms = (time.perf_counter_ns() - start) / 10**6

        try:
            client.bots.make_move(self.game_id, move)
        except berserk.exceptions.ResponseError as e:  # you flagged
            print(e)
            return

        print(f"time: {time_spent_ms:.0f} - kns: {self.bot.nodes / time_spent_ms:.0f}")
        # print(f"time delta: {time_spent_ms - time_limit:.0f} ms")
        print("-" * 40)

    def look_in_da_book(self):
        fruit = chess.polyglot.open_reader("book/book_small.bin")
        if fruit.get(self.pcb):
            return fruit.weighted_choice(self.pcb)

    def syzygy(self):
        html_fen = self.pcb.fen().replace(" ", "_")
        response = requests.get(f"http://tablebase.lichess.ovh/standard?fen={html_fen}").json()
        return response['moves'][0]


print("id name black_numba")
print("id author Avo-k")
print("compiling...")
compiling_time = time.time()
search(Black_numba(), parse_fen(start_position), print_info=False, depth_limit=2)
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
