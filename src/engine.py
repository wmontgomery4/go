import itertools
import numpy as np

EMPTY = 0
BLACK = 1
WHITE = -1
COLUMNS = "ABCDEFGHJKLMNOPQRST"
PASS = ()


class Engine():
    def __init__(self, size=19, komi=6.5):
        self.size = size
        self.komi = komi
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.last_move = None
        self.ko = None

    # TODO: Fix ko rule.
    # TODO: Suicide rule?
    def legal(self, move, color):
        if move == self.ko:
            return False
        elif self.board[move] != EMPTY:
            return False
        return True

    def make_move(self, move, color):
        assert self.legal(move, color)
        self.last_move = move
        if move == PASS:
            return
        # 1) Place stone
        self.board[move] = color
        # 2) Clear opponent
        for p in self._neighbors(move):
            if self.board[p] == -color:
                self._clear(p)
        # 3) Clear self
        self._clear(move)

    def score(self):
        score = self.board.sum() - self.komi
        counted = set()
        for position in itertools.product(range(self.size), repeat=2):
            color = self.board[position]
            if color != EMPTY or position in counted:
                continue
            visited = self._flood(position)
            if not (visited[BLACK] and visited[WHITE]):
                score += len(visited[EMPTY]) * (BLACK if visited[BLACK] else WHITE)
            counted.update(visited[color])
        return score

    def _clear(self, position):
        color = self.board[position]
        visited = self._flood(position)
        if not visited[EMPTY]:
            for p in visited[color]:
                self.board[p] = EMPTY

    def _flood(self, position, visited=None):
        if not visited:
            visited = {EMPTY:set(), BLACK:set(), WHITE:set()}
        color = self.board[position]
        visited[color].add(position)
        for n in self._neighbors(position):
            n_color = self.board[n]
            if n_color != color:
                visited[n_color].add(n)
            elif n not in visited[color]:
                self._flood(n, visited)
        return visited

    def _neighbors(self, position):
        r, c = position
        neighbors = []
        if r > 0:
            neighbors.append((r-1, c))
        if r < self.size - 1:
            neighbors.append((r+1, c))
        if c > 0:
            neighbors.append((r, c-1))
        if c < self.size - 1:
            neighbors.append((r, c+1))
        return neighbors

    def string_from_move(self, move):
        row, col = move
        number = str(self.size - row)
        letter = COLUMNS[col]
        return letter + number

    def move_from_string(self, string):
        letter = string[0].upper()
        number = string[1:]
        row = self.size - int(number)
        col = ord(letter) - ord('A')
        if ord(letter) > ord('I'):
            col -= 1
        return (row, col)

    def __str__(self):
        """ Render board state as string. Not very efficient. """
        # Initialize char grid.
        grid = np.empty((self.size, self.size), dtype=str)
        grid[:]= '+'
        # Corners.
        grid[0, 0] = '\u250C'
        grid[0, -1] = '\u2510'
        grid[-1, 0] = '\u2514'
        grid[-1, -1] = '\u2518'
        # Sides.
        grid[1:-1, 0] = '\u251C'
        grid[0, 1:-1] = '\u252C'
        grid[1:-1, -1] = '\u2524'
        grid[-1, 1:-1] = '\u2534'
        # Star points.
        if self.size == 9:
            grid[4,4] = '\u25CD'
            grid[2::4, 2::4] = '\u25CD'
        elif self.size == 19:
            grid[3:16:6, 3:16:6] = '\u25CD'
        # Stones.
        grid[self.board == BLACK] = '\u25EF'
        grid[self.board == WHITE] = '\u2B24'
        # Build the string.
        cols = COLUMNS[:self.size]
        string = "   " + " ".join(list(cols))
        for i, row in enumerate(grid):
            string += "\n{:2d} ".format(self.size-i)
            string += " ".join(row)
            string += " {:d}".format(self.size-i)
        string += "\n   " + " ".join(list(cols))
        return string


if __name__ == '__main__':
    engine = Engine()
    # Capture a stone
    engine.make_move(engine.move_from_string("D4"), WHITE)
    engine.make_move(engine.move_from_string("D3"), BLACK)
    engine.make_move(engine.move_from_string("C4"), BLACK)
    engine.make_move(engine.move_from_string("E4"), BLACK)
    engine.make_move(engine.move_from_string("D5"), BLACK)
    # Add an extra stone so that black doesn't get all points
    engine.make_move(engine.move_from_string("Q4"), WHITE)
    print(engine)
    print("Score (should be -2.5): ", engine.score())
