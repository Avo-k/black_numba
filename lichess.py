import berserk
import time

from position import parse_fen
from constants import start_position
from search import Black_numba, search, random_move
from move_gen import get_move_uci, make_move
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
        self.deltas = []

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
        t = game_state["wtime" if self.bot_is_white else "btime"]
        remaining_time = t / 1000 if isinstance(t, int) else t.minute * 60 + t.second

        # Set limits
        time_limit = remaining_time / 60
        depth_limit = max(3, remaining_time // 6)
        nodes_limit = time_limit * 100000

        # look for a move
        move = depth = score = None
        for depth, move, score in search(self.bot, self.pos, print_info=False):
            if time.time() - start > time_limit:
                break
            if depth == depth_limit:
                break
            if self.bot.nodes > nodes_limit:
                break

        actual_time = time.time() - start

        try:
            client.bots.make_move(self.game_id, get_move_uci(move))
        except:  # you loose
            return

        self.pos = make_move(self.pos, move)
        self.moves += " " + get_move_uci(move)

        self.deltas.append(actual_time - time_limit)
        print("-" * 40)
        print(f"depth: {depth} - time: {round(actual_time, 2)} seconds")
        print(f"score: {score} - time delta: {round(actual_time - time_limit, 2)}")
        print(f"nodes: {self.bot.nodes} - n/s: {round(self.bot.nodes / actual_time)}")
        print(f"deltas means {round(sum(self.deltas) / len(self.deltas), 2)}")

    def make_first_move(self):
        move = depth = score = None
        for depth, move, score in search(self.bot, self.pos, print_info=False):
            if depth == 4:
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
        print("new game")
        game_id = event['game']['id']
        game = Game(client=client, game_id=game_id)
        game.run()

    else:  # challengeDeclined, gameFinish, challengeCanceled
        if event['type'] not in ('challengeDeclined', 'gameFinish', 'challengeCanceled'):
            print('NEW EVENT', event)
