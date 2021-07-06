from multiprocessing import Pool
from eval_network import *
import random
from board_conversion import *
import numpy as np
import chess
import time

eval_model = define_evaluator()
board = chess.Board()
epochs = 10

def evaluate(x):
    return eval_model.predict(x.reshape(1,8,8,12)).flatten()[0]

def create_mp(board):
    legal_moves = str(board.legal_moves)[38:-2].replace(',','').split()
    mp_array = np.array([(board,legal_move) for legal_move in legal_moves])
    return mp_array
    
def random_path(board,move,depth = 10):
    play_board = board.copy()
    play_board.push_san(move)
    for _ in range(depth):
        legal_moves = str(play_board.legal_moves)[38:-2].replace(',','').split()
        if legal_moves:
            move = random.choice(legal_moves)
            play_board.push_san(move)
        else:
            break
    matrix = make_matrix(play_board.copy())
    translated = np.array(translate(matrix))
    return evaluate(translated)

def eval_mp(board):
    mp_array = create_mp(board)
    time1 = time.time()
    with Pool(5) as p2:
        eval = p2.starmap(random_path,mp_array)
    p2.close()
    time2 = time.time()
    return eval
'''
if __name__ == '__main__':
    time1 = time.time()
    for epoch in range(epochs):
        print('EPOCH:',epoch)
        if epoch == 0:
            score = np.array(eval_mp(board))
        else:
            score += np.array(eval_mp(board))
        print(score)
    time2 = time.time()
    print(time2-time1)
'''

time1 = time.time()
random_path(chess.Board(),'e4')
time2 = time.time()
print(time2-time1)
    
# Time taken for depth 10, 10 epochs: 217.97s 
        

