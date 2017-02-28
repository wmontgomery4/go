import numpy as np

EMPTY = 0
BLACK = 1
WHITE = -1
COLUMNS = "ABCDEFGHJKLMNOPQRST"

class Engine():
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.libs = np.zeros((self.size, self.size), dtype=int)
        self.last_move = None
        self.ko = None

    def gen_move(self, color):
        # TODO: neural net stuff.
        while True:
            move = tuple(np.random.randint(self.size, size=2))
            if self.legal(move, color):
                return move

    def legal(self, move, color):
        if move is self.ko:
            return False
        if self.board[move] != EMPTY:
            return False
        # TODO:
        return True

    def play(self, move, color):
        assert self.legal(move, color)
        self.board[move] = color
        self.ko = self.last_move
        self.last_move = move

    def move_from_string(self, string):
        # TODO: error handling
        letter = string[0]
        number = string[1:]
        row = self.size - int(number)
        col = ord(letter) - ord('A')
        if ord(letter) > ord('I'):
            col -= 1
        return (row, col)

    def __unicode__(self):
        """ Render board state as string. Not very efficient. """
        # Initialize char grid.
        grid = np.empty((self.size, self.size), dtype=unicode)
        grid[:]= u'+'
        # Corners.
        grid[0, 0] = u'\u250C'
        grid[0, -1] = u'\u2510'
        grid[-1, 0] = u'\u2514'
        grid[-1, -1] = u'\u2518'
        # Sides.
        grid[1:-1, 0] = u'\u251C'
        grid[0, 1:-1] = u'\u252C'
        grid[1:-1, -1] = u'\u2524'
        grid[-1, 1:-1] = u'\u2534'
        # Stones.
        grid[self.board < 0] = u'\u26AA'
        grid[self.board > 0] = u'\u26AB'
        # Build the string.
        cols = COLUMNS[:self.size]
        string = "   " + " ".join(list(cols))
        for i, row in enumerate(grid):
            string += "\n{:2d} ".format(self.size-i)
            string += " ".join(row)
            string += " {:d}".format(self.size-i)
        string += "\n   " + " ".join(list(cols))
        return string

if __name__ == "__main__":
    engine = Engine()
    while True:
        print unicode(engine)
        # Player picks a move.
        string = raw_input('\nMove: ')
        engine.play(engine.move_from_string(string), BLACK)
        # Bot responds.
        move = engine.gen_move(WHITE)
        engine.play(move, WHITE)
