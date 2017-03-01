from engine import *
from IPython import embed

class CLI():
    def __init__(self):
        self.query = "\nLast move: {}\nYour move: "

    def gen_move(self, engine, color):
        prev = engine.prev_move
        prev = "" if not prev else engine.string_from_move(prev)
        while True:
            print unicode(engine)
            string = raw_input(self.query.format(prev))
            if string == 'debug':
                embed()
            if string == '':
                return PASS
            try:
                move = engine.move_from_string(string)
                assert engine.legal(move, color)
                return move
            except:
                print "Illegal move! Try again."

def rollout(black, white, size=19):
    engine = Engine(size)
    while True:
        # Black plays a move.
        move = black.gen_move(engine, BLACK)
        engine.play(move, BLACK)
        # White plays a move.
        move = white.gen_move(engine, WHITE)
        engine.play(move, WHITE)
        # Check if both players passed.
        if engine.ko == engine.prev_move:
            return engine


if __name__ == "__main__":
    from bot import Bot
    bot = Bot(size=9)
    human = CLI()
    engine = rollout(human, bot, size=9)
    print "Score: {}".format(engine.score())
    embed()
