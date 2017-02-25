import numpy as np

BLACK = 1
WHITE = -1

class Board():
    def __init__(self, size=19):
        self.size = size
        self.grid = np.zeros((self.size,self.size), dtype=int)

    def play(self, move, color):
        self.grid[move] = color

    def __getitem__(self, key):
        return self.grid[key]

    def __setitem__(self, key, value):
        self.grid[key] = value

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
        grid[1:-1, 0] = u"\u251C"
        grid[0, 1:-1] = u"\u252C"
        grid[1:-1, -1] = u"\u2524"
        grid[-1, 1:-1] = u"\u2534"
        # Stones.
        grid[self.grid < 0] = u'\u26AA'
        grid[self.grid > 0] = u'\u26AB'

        cols = "ABCDEFGHJKLMNOPQRST"[:self.size]
        string = "  " + " ".join(list(cols))
        for i, row in enumerate(grid):
            string += "\n{:2d}".format(i+1)
            string += " ".join(row)
        return string

if __name__ == "__main__":
    board = Board()
    X, Y = np.mgrid[3:15:3j,3:15:3j]
    board[X.astype(int), Y.astype(int)] = 1
    board[2, 5] = -1
    print unicode(board)
