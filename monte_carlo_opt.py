import random
from board_conversion import *
import numpy as np
import time 

def monte_carlo_algo(board,evaluation,epochs = 5,depth = 5):
    first_legal_moves = list(board.legal_moves)
    scores = np.ones(len(first_legal_moves))
    for epoch in range(epochs):

        for first_move in range(len(first_legal_moves)):
            play_board = board.copy()
            play_board.push(first_legal_moves[first_move])

            for _ in range(depth):
                legal_moves = list(play_board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    play_board.push(move)
                else:
                    break
                
            translated = np.array(translate(play_board))
            scores[first_move] += evaluation(translated)
    idx = np.where(scores == max(scores))[0][0]
    return first_legal_moves[idx]