import chess
from eval_network import *
from monte_carlo import *

def fitness(agents):
    for agent in range(len(agents)-1):
        game = []
        board = chess.Board()
        
        other_agents = agents
        other_agents.remove(agents[agent])
        player_2 = random.choice(other_agents)
        
        player_1_idx = agent
        player_2_idx = agents.index(player_2)
        
        player_1 = agents[agent]

        counter = 0
        print('Game Started between Agent',player_1_idx,'and Agent',player_2_idx)
        while not(board.is_game_over()):
            print('Move '+str(counter)+'...')

            if counter % 2 == 0:
                model = player_1.neural_network
            else:
                model = player_2.neural_network

            def evaluation(input):
                pred = model.predict(input.reshape(1,8,8,12))
                return pred

            move = calculate_move(board,evaluation,epochs = 10,depth = 10)
            board.push_san(move)
            game.append(move)
            counter += 1

        agents[player_1_idx].game = game
        agents[player_2_idx].game = game

        if board.is_checkmate:
            if counter % 2 == 0:
                agents[player_1_idx].fitness *= 1.5
                agents[player_2_idx].fitness *= 0.8
            else:
                agents[player_2_idx].fitness *= 1.5
                agents[player_1_idx].fitness *= 0.8
    return agents


