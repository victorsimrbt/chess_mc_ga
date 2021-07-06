import random
from board_conversion import *
import numpy as np

def calculate_move(board,evaluation,epochs = 10,depth = 10):
    first_legal_moves = str(board.legal_moves)[38:-2].replace(',','').split()
    scores = [[0]] * len(first_legal_moves)
    for epoch in range(epochs):
        for first_move in range(len(first_legal_moves)):
            play_board = board.copy()
            play_board.push_san(first_legal_moves[first_move])
            for _ in range(depth):
                legal_moves = str(play_board.legal_moves)[38:-2].replace(',','').split()
                if legal_moves:
                    move = random.choice(legal_moves)
                    play_board.push_san(move)
                else:
                    break
            matrix = make_matrix(play_board.copy())
            translated = np.array(translate(matrix))
            scores[first_move] += evaluation(translated)
    return first_legal_moves[scores.index(max(scores))]