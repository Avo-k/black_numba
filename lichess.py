import berserk
import time
import random
import threading

from position import parse_fen, print_position
from constants import start_position
from search import Black_numba, random_move
from move_gen import get_move_uci, make_move
from uci import parse_move
import chess.polyglot

API_TOKEN = open("api_token.txt").read()
bot_id = 'black_numba'

session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)


class Game(threading.Thread):
    def __init__(self, client, game_id, **kwargs):
        super().__init__(**kwargs)
        self.game_id = game_id
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        self.current_state = next(self.stream)
        self.bot_is_white = self.current_state['white'].get('id') == bot_id
        self.pos = parse_fen(start_position)
        self.moves = ""
        self.bot = Black_numba()

    def run(self):
        print("game start")

        # From Position
        if self.current_state['variant']['short'] == "FEN":
            self.pos = parse_fen(self.current_state['initialFen'])
            if not self.pos.side:
                self.make_first_move()

        # Reconnect to a game
        elif self.moves != self.current_state['state']['moves']:

            move_list = self.current_state['state']['moves'].split()

            for smove in move_list[:-1]:
                move = parse_move(self.pos, smove)
                self.pos = make_move(self.pos, move)

            self.make_first_move()

        # if you have to start
        elif not self.current_state['state']['moves'] and self.bot_is_white:
            self.make_first_move()

        # main game loop
        for event in self.stream:
            if event['type'] == 'gameState':
                if event['status'] == 'started':
                    m = self.handle_state_change(event)
                    if m == "checkmate":
                        break
                elif event['status'] in ('mate', 'resign', 'outoftime', 'aborted', 'draw'):
                    break
                else:
                    print('NEW', event['status'])
                    break
            elif event['type'] == 'chatLine':
                self.handle_chat_line(event)

    def handle_state_change(self, game_state):
        if game_state['moves'] == self.moves:
            return
        self.moves = game_state['moves']
        new_move = parse_move(self.pos, self.moves.split()[-1])
        self.pos = make_move(self.pos, new_move)

        s = time.time()

        move = self.bot.search(self.pos, 4)

        print("time needed:", round(time.time() - s, 2))

        if move is None:
            return "checkmate"
        client.bots.make_move(self.game_id, get_move_uci(move))
        self.pos = make_move(self.pos, move)
        self.moves += " " + get_move_uci(move)

    def make_first_move(self):
        move = self.bot.search(self.pos, 4)
        client.bots.make_move(self.game_id, get_move_uci(move))
        self.pos = make_move(self.pos, move)
        self.moves += get_move_uci(move)

    def handle_chat_line(self, chat_line):
        pass


print("ready to play")
for event in client.bots.stream_incoming_events():
    if event['type'] == 'challenge':
        client.bots.accept_challenge(event['challenge']['id'])
    elif event['type'] == 'gameStart':
        game = Game(client, event['game']['id'])
        game.run()
