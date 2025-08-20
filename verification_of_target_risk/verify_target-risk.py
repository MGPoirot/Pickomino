from utils import json_in, json_out
from precalculate_turn_chances import present_roll_options, get_turn_probabilities, Dice, Tiles
import numpy as np
import pandas as pd
from pathlib import Path
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
Run dice roll simulations to verify that the probabilities
we have calculated for dice roll outcomes can be achieved.
"""


def run_simulation(f_name, evals, start_arr, min_chance):
    tiles_collected = []
    for i in range(evals):
        # Start with a clean slate
        collection = start_arr.copy()

        # Loop until 1) no dice to pick up, 2) no dice to roll or 3) target reached
        while True:
            # Start the turn with a roll of the dice
            roll = collection.roll()

            # Look up the odds of reaching our target score
            chance_chart = present_roll_options(collection, roll)

            if not any(chance_chart):
                # 1) We only rolled dice that we cannot pick up and lost
                tiles_collected.append(0)
                break

            mask = ((chance_chart > min_chance).sum(1) > 0)
            if any(mask):
                idx = int(chance_chart[mask].iloc[-1].idxmax()[0]) - 1
            else:
                # We are beggars and cant choose an option that suits our risk profile
                # We assume we have go for reaching the 21 tile and shoot our best shot
                idx = int((chance_chart - min_chance).iloc[0].idxmax()[0]) - 1

            # Insert the dice into the dice
            collection[idx] = roll[idx]

            forced_exit = not collection.free_dice or not collection.free_faces

            # Evaluate
            chance_chart = get_turn_probabilities(collection)
            roll_chance = chance_chart[get_turn_probabilities(collection) != 1]
            score = collection.score
            score = score if score >= np.min(Tiles.values) else 0
            voluntary_exit = roll_chance.max() <= min_chance and score > 0

            # 3) Exit the game for one of three reasons
            if forced_exit or voluntary_exit:
                tiles_collected.append(score)
                break
    json_out(tiles_collected, f_name)


def collect_simulation_results(risk, evals=10_000):
    min_chance = 1 - risk

    start_arr = np.array((0, 0, 0, 0, 0, 0)).view(Dice)
    f_name = Path('') / f'collected-tiles_evals-{evals}_target-risk_{risk:.2f}.json'
    # if not f_name.is_file():
    #    run_simulation(f_name, evals, start_arr, min_chance)
    tiles_collected = json_in(f_name)
    # Compute mean and SD over all runs
    win_rate = np.count_nonzero(tiles_collected) / evals
    mean = np.mean(tiles_collected)
    mean_s = np.mean([t for t in tiles_collected if t])
    mean_sd = np.std([t for t in tiles_collected if t])
    return win_rate, mean, mean_s, mean_sd


if __name__ == '__main__':
    # Print a data frame containing our simulation outcomes
    print(pd.DataFrame.from_dict(
        data={risk: collect_simulation_results(risk, evals=100_000) for risk in np.linspace(0, 1, 101)},
        orient='index',
        columns=[
            'win_rate',
            'mean_score',
            'mean_score_of_successes',
            'sd_of_successes',
        ],
    ).to_string(
        formatters={
            'win_rate': '{:,.1%}'.format,
        }
    ))
