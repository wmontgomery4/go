import numpy as np

EMPTY = 0
BLACK = 1
WHITE = -1
COLUMNS = "ABCDEFGHJKLMNOPQRST"

from bot import Bot

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
        # TODO: Try other data structures, compare speed.
        assert self.legal(move, color)
        # Place the stone and update flags.
        self.board[move] = color
        self.ko = self.last_move
        self.last_move = move
        # Flood the loss of liberties, keep track of uncounted stones.
        visited = np.zeros((self.size, self.size), dtype=bool)
        affected = set([move])
        self._flood(self._first_sweep, move, color, visited, affected)
        # Count all of the remaining groups.
        all_visited = np.zeros((self.size, self.size), dtype=bool)
        for position in affected:
            if all_visited[position]:
                continue
            visited = np.zeros((self.size, self.size), dtype=bool)
            counted = set()
            self._count(position, color, visited, counted)
            self.libs[visited] = len(counted)
            all_visited += visited

    def _flood(self, fn, position, *args):
        r, c = position
        if r > 0:
            fn((r-1, c), *args)
        if r < self.size - 1:
            fn((r+1, c), *args)
        if c > 0:
            fn((r, c-1), *args)
        if c < self.size - 1:
            fn((r, c+1), *args)

    def _first_sweep(self, position, color, visited, affected):
        if visited[position] or self.board[position] != -color:
            return
        if self.libs[position] > 1:
            self._reduce(position, -color, visited)
        else:
            self._remove(position, -color, visited, affected)

    def _reduce(self, position, color, visited):
        if visited[position] or self.board[position] != color:
            return
        visited[position] = True
        self.libs[position] -= 1
        self._flood(self._reduce, position, color, visited)

    def _remove(self, position, color, visited, affected):
        if visited[position] or self.board[position] == EMPTY:
            return
        if self.board[position] == -color:
            affected.add(position)
            return
        visited[position] = True
        self.board[position] = EMPTY
        self.libs[position] = 0
        self._flood(self._remove, position, color, visited, affected)

    def _count(self, position, color, visited, counted):
        if visited[position] or self.board[position] == -color:
            return
        if self.board[position] == EMPTY:
            counted.add(position)
            return
        visited[position] = True
        self._flood(self._count, position, color, visited, counted)

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
    bot = Bot(engine, WHITE)
    while True:
        print unicode(engine)
        # Player picks a move.
        string = raw_input('\nMove: ')
        if string == 'libs':
            # For debugging.
            print engine.libs
            continue
        try:
            engine.play(engine.move_from_string(string), BLACK)
        except:
            print "Illegal move! Try again."
            continue
        # Bot responds.
        move = bot.act()
        engine.play(move, WHITE)

if __name__ == "__main__":
    interactive_session()

#    engine = Engine()
#    engine.play(engine.move_from_string("D4"), WHITE)
#    engine.play(engine.move_from_string("D5"), WHITE)
#    engine.play(engine.move_from_string("D3"), BLACK)
#    engine.play(engine.move_from_string("C4"), BLACK)
#    engine.play(engine.move_from_string("C5"), BLACK)
#    engine.play(engine.move_from_string("E4"), BLACK)
#    engine.play(engine.move_from_string("E5"), BLACK)
#    print unicode(engine)
