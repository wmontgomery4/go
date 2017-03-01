from engine import *
from bot import Bot

def gen_move(engine, color):
    prev = engine.string_from_move(engine.prev_move)
    query = "\nLast move: {}\nYour move: ".format(prev)
    while True:
        print unicode(engine)
        string = raw_input(query)
        if string == 'debug':
            from IPython import embed; embed()
        try:
            move = engine.move_from_string(string)
            assert engine.legal(move, color)
            return move
        except:
            print "Illegal move! Try again."
            continue

def interactive_session():
    engine = Engine()
    bot = Bot()
    while True:
        # Bot plays black.
        bot.act(engine, BLACK)
        # Human plays white.
        move = gen_move(engine, WHITE)
        engine.play(move, WHITE)

if __name__ == "__main__":
    interactive_session()
