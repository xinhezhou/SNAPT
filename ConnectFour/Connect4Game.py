import numpy as np


class C4Game():
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """

    def __init__(self, height=6, width=7, win_length=4):
        self.height = height
        self.width = width
        self.win_length = win_length
        self.init_board = np.zeros((height, width), dtype = int)

    def getInitBoard(self):
        return self.init_board

    def getBoardSize(self):
        return (self.height, self.width)

    def getActionSize(self):
        return self.width

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""
        new_board = np.copy(board) 
        for k in reversed(range(self.height)):
            if new_board[k, action] == 0:
                new_board[k, action] = player
                return new_board, -player
        return new_board, -player

    def getValidMoves(self, board, player):
        "Any zero value in top row in a valid move"
        f = lambda x: 1 if x == 0 else 0
        f = np.vectorize(f)
        return f(board[0])

    def getGameEnded(self, board, player):
        "Horizontal check"
        for h in range(self.height):
            for w in range(self.width - self.win_length + 1):
                s = np.sum(board[h, w:w +self.win_length])
                # the player returning is kind of unintuitive but just get used to it
                if s == self.win_length:
                    return player
                if s == -self.win_length:
                    return -player
                
        "Vertical check"
        for h in range(self.height - self.win_length + 1):
            for w in range(self.width):
                s = np.sum(board[h: h + self.win_length, w])
                if s == self.win_length:
                    return player
                if s == -self.win_length:
                    return -player
        
        "Diagonal check"
        for h in range(self.height - self.win_length + 1):
            for w in range(self.width - self.win_length + 1):
                forward = 0
                backward = 0
                for k in range(self.win_length):
                    forward += board[h + k, w + k]
                    backward += board[self.height - 1 - h - k, w + k]
                    
                if forward == self.win_length or backward == self.win_length:
                    return player
                if forward == -self.win_length or backward == -self.win_length:
                    return -player
                
        "tie check"
        if np.sum(self.getValidMoves(board, player)) == 0:
            return 1e-4
        return 0
                
                

    def getCanonicalForm(self, board, player):
        # Flip player from 1 to -1
        return board * player

    def getSymmetries(self, board, pi):
        """Board is left/right board symmetric"""
        return [(board, pi), (board[:, ::-1], pi[::-1])]

    def stringRepresentation(self, board):
        return str(board)

    @staticmethod
    def display(board):
        print(str(board))



