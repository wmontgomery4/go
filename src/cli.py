from IPython import embed
from engine import *


class CLI():
    def __init__(self):
        self.query = "\nLast move: {}\nYour move: "

    def gen_move(self, engine, color):
        last = engine.last_move
        last = "" if not last else engine.string_from_move(last)
        while True:
            print(engine)
            string = input(self.query.format(last))
            if string == 'debug':
                embed()
            elif string == '':
                return PASS
            try:
                move = engine.move_from_string(string)
                assert engine.legal(move, color)
                return move
            except:
                print("Illegal move! Try again.")
