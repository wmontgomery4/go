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
        # TODO: keep track of prisoners?

    def legal(self, move, color):
        if self.board[move] != EMPTY:
            return False
        if move is self.ko:
            return False
        # TODO: Suicide rule.
        return True

    def play(self, move, color):
        assert self.legal(move, color)
        # Place the stone and update flags.
        self.board[move] = color
        self.libs[move] = 4
        self.ko = self.last_move
        self.last_move = move
        # Flood the loss of liberties.
        self._flood(self._reduce_libs, move, -color)

    def _flood(self, fn, position, color):
        visited = np.zeros((self.size, self.size), dtype=bool)
        visited[position] = True
        r, c = position
        if r > 0:
            fn((r-1, c), color, visited)
        if r < self.size - 1:
            fn((r+1, c), color, visited)
        if c > 0:
            fn((r, c-1), color, visited)
        if c < self.size - 1:
            fn((r, c+1), color, visited)

    def _reduce_libs(self, position, color, visited):
        """ Reduce liberties of opposite color. """
        if visited[position] or self.board[position] != color:
            return
        visited[position] = True
        # Reduce liberties.
        self.libs[position] -= 1
        # Remove dead stones.
        if self.libs[position] == 0:
            self.board[position] = EMPTY
        # Flood forward.
        r, c = position
        if r > 0:
            self._reduce_libs((r-1, c), color, visited)
        if r < self.size - 1:
            self._reduce_libs((r+1, c), color, visited)
        if c > 0:
            self._reduce_libs((r, c-1), color, visited)
        if c < self.size - 1:
            self._reduce_libs((r, c+1), color, visited)

    def _count_libs(self, move, color, visited, libs):
        if self.board[move] == EMPTY:
            libs[move] = True
            return
        r, c = move
        visited = np.zeros((self.size, self.size), dtype=bool)
        libs = np.zeros((self.size, self.size), dtype=bool)

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
        # Star points.
        if self.size == 19:
            grid[3:16:6, 3:16:6] = u'\u25CD'
        # Stones.
        grid[self.board == BLACK] = u'\u25EF'
        grid[self.board == WHITE] = u'\u2B24'
        # Build the string.
        cols = COLUMNS[:self.size]
        string = "   " + " ".join(list(cols))
        for i, row in enumerate(grid):
            string += "\n{:2d} ".format(self.size-i)
            string += " ".join(row)
            string += " {:d}".format(self.size-i)
        string += "\n   " + " ".join(list(cols))
        return string

def interactive_session():
    engine = Engine()
    from bot import Bot
    bot = Bot(engine, WHITE)
    while True:
        print unicode(engine)
        # Player picks a move.
        string = raw_input('\nMove: ')
        engine.play(engine.move_from_string(string), BLACK)
        # Bot responds.
        move = bot.act()
        engine.play(move, WHITE)

if __name__ == "__main__":
#    interactive_session()

    engine = Engine()
    engine.play(engine.move_from_string("D4"), WHITE)
    engine.play(engine.move_from_string("C4"), BLACK)
    engine.play(engine.move_from_string("D3"), BLACK)
    engine.play(engine.move_from_string("E4"), BLACK)
    engine.play(engine.move_from_string("D5"), BLACK)
    print unicode(engine)
