import numpy as np
import torch
import random
from Othello.Arena import Arena
from Othello.OthelloGame import OthelloGame
from Othello.OthelloPlayers import *
from MCTS import Coach

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    game = OthelloGame(5)  # An Othello game with a 5*5 board
    random_player = RandomPlayer(game).play
    coach = Coach(game)
    coach.train()
    print("\nTESTING")
    arena = Arena(coach.play, random_player, game)
    oneWon, twoWon, draws = arena.playGames(100)
    fraction_won = oneWon / 100
    print("Fractin won: ", fraction_won)