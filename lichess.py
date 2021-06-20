import berserk
import time
import sys

from position import parse_fen, print_position
from constants import start_position, LOWER_MATE
from search import Black_numba, search, random_move
from moves import get_move_uci, make_move
from uci import parse_move

API_TOKEN = open("api_token.txt").read()
bot_id = 'black_numba'

session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session=session)


class Game:
    def __init__(self, client, game_id):
        # super().__init__(**kwargs)
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
            for smove in move_list:
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
                    self.handle_state_change(event)
                    # self.play_random_fast(event)
                elif event['status'] in ('mate', 'resign', 'outoftime', 'aborted', 'draw'):
                    break
                else:
                    print('NEW', event['status'])
                    break

    def play_random_fast(self, game_state):
        if game_state['moves'] == self.moves:
            return
        self.moves = game_state['moves']
        new_move = parse_move(self.pos, self.moves.split()[-1])
        self.pos = make_move(self.pos, new_move)

        move = random_move(self.pos)
        client.bots.make_move(self.game_id, get_move_uci(move))

        self.pos = make_move(self.pos, move)
        self.moves += " " + get_move_uci(move)

    def handle_state_change(self, game_state):
        if game_state['moves'] == self.moves:
            return
        self.moves = game_state['moves']
        new_move = parse_move(self.pos, self.moves.split()[-1])
        self.pos = make_move(self.pos, new_move)

        # set time variables
        start = time.time()
        t, opp_t = (game_state["wtime"], game_state["btime"]) if self.bot_is_white else (
        game_state["btime"], game_state["wtime"])

        def in_sec(tim):
            return tim / 1000 if isinstance(tim, int) else tim.minute * 60 + tim.second

        remaining_time = in_sec(t)
        remaining_opp_t = in_sec(opp_t)

        # Set limits
        time_limit = remaining_time / 60
        nodes_limit = time_limit * 100000

        # look for a move
        move, depth = None, None
        for depth, move, score in search(self.bot, self.pos, print_info=True):
            if time.time() - start > time_limit:
                break
            if self.bot.nodes > nodes_limit:
                break
            if score > LOWER_MATE:
                break

        actual_time = time.time() - start + 0.001

        try:
            client.bots.make_move(self.game_id, get_move_uci(move))
        except:  # you loose
            return

        self.pos = make_move(self.pos, move)
        self.moves += " " + get_move_uci(move)

        print(f"time: {actual_time:.2f} - kns: {self.bot.nodes / actual_time / 1000:.0f}")

        # pondering
        ponder_limit = min(time_limit / 2.5, remaining_opp_t / 150)
        ponder_start = time.time()
        ponder_depth = 0
        for ponder_depth, _, _ in search(self.bot, self.pos, print_info=False):
            if time.time() - ponder_start > ponder_limit or ponder_depth >= depth - 1:
                break
        print("-" * 12)
        print(f"{ponder_depth=} in {time.time() - ponder_start:.2f} sec")
        print("-" * 40)

    def make_first_move(self):
        move = None
        for depth, move, score in search(self.bot, self.pos, print_info=True):
            if depth == 8:
                break
        client.bots.make_move(self.game_id, get_move_uci(move))
        self.pos = make_move(self.pos, move)
        self.moves += get_move_uci(move)


print("id name black_numba")
print("id author Avo-k")
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
        game_id = event['game']['id']
        game = Game(client=client, game_id=game_id)
        game.run()
        del game

    else:  # challengeDeclined, gameFinish, challengeCanceled
        if event['type'] not in ('challengeDeclined', 'gameFinish', 'challengeCanceled'):
            print('NEW EVENT', event)
