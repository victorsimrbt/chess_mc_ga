from fitness import *
from genetic_algo import *
from eval_network import *

eval_model = define_evaluator()
tourno_fitness = fitness

ga = genetic_algorithm()
agent,loss = ga.execute(tourno_fitness,eval_model,pop_size = 10,generations = 10)
print(agent.fitness)

board = chess.Board()

for move in agent.game:
    board.push_san(move)
print(board)
print(board.is_checkmate())
print(agent.game)