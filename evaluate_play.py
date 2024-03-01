from utils import Tiles, Dice, Player
from evaluate_turn import probabilities
from itertools import product
import numpy as np
import pandas as pd

turn_probabilities = probabilities[Dice().key]

tile_states = product(range(2), repeat=Tiles.n_tiles)

player1 = Player()
player2 = Player()

game_state = {}


def get_play_probabilities(tiles: Tiles, player: Player) -> pd.Series:
    if tiles.key in probabilities:
        return probabilities[tiles.key]
    if not tiles.free_tiles:
        probabilities[tiles.key] = (player.position > 0)




for ts in tile_states:
    tiles = Tiles(ts)
    break
    boundary = 10
    max_diff = np.min((boundary, np.sum([v for s, v in zip(ts, Tiles.values.values()) if not s])))
    for score_diff in range(-max_diff, max_diff + 1):
        if not tiles.free_tiles:
            game_state[score_diff, tiles.key] = float(score_diff > 0)
            if tiles.n_free_tiles == 1:
                [i for i, j in zip(turn_probabilities, tiles) if j]
