import sys
from engine import *
from IPython import embed

class CLI():
    def __init__(self):
        self.query = "\nLast move: {}\nYour move: "

    def gen_move(self, engine, color):
        last = engine.last_move
        last = "" if not last else engine.string_from_move(last)
        while True:
            print unicode(engine)
            string = raw_input(self.query.format(last))
            if string == 'debug':
                embed()
            elif string == '':
                return PASS
            try:
                move = engine.move_from_string(string)
                assert engine.legal(move, color)
                return move
            except:
                print "Illegal move! Try again."

def rollout(engine, black, white, moves=500):
    for i in range(moves // 2):
        # Black plays a move.
        move = black.gen_move(engine, BLACK)
        if move == engine.last_move == PASS:
            break
        engine.make_move(move, BLACK)
        # White plays a move.
        move = white.gen_move(engine, WHITE)
        if move == engine.last_move == PASS:
            break
        engine.make_move(move, WHITE)
    return engine


if __name__ == "__main__":
    human = CLI()
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    from bot import Bot
    bot = Bot(size)
    engine = Engine(size)
    rollout(engine, human, bot)
    print "Score: {}".format(engine.score())
    embed()
