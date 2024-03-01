from utils import json_in, json_out, Tiles, Dice
from scipy.stats import norm
from evaluate_turn import present_roll_options, get_turn_probabilities
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


def run_simulation(f_name, evals, start_arr, target_score):
    tiles_collected = []
    for i in range(evals):
        # Start with a clean slate
        collection = start_arr.copy()

        # Loop until 1) no dice to pick up, 2) no dice to roll or 3) target reached
        while True:
            # Start the turn with a roll of the dice
            roll = collection.roll()

            # Look up the odds of reaching our target score
            info = present_roll_options(collection, roll)

            if not any(info):
                # 1) We only rolled dice that we cannot pick up and lost
                tiles_collected.append(0)
                break

            # Get the face value with the highest chance of reaching our target score
            idx = int(info.loc[target_score].idxmax(axis=1).values[0][0]) - 1

            # Insert the dice into the dice
            collection[idx] = roll[idx]

            if collection.score >= target_score:
                # 3) We have achieved the desired score and won
                tiles_collected.append(collection.score)
                break
            elif not collection.free_faces or not collection.free_dice:
                # 1,2) All faces or all dice have been collected
                tiles_collected.append(0)
                break
    json_out(tiles_collected, f_name)


def collect_simulation_results(target_score, evals=10_000, alpha=0.05):
    start_arr = np.array((0, 0, 0, 0, 0, 0)).view(Dice)

    expectation = get_turn_probabilities(start_arr).loc[target_score]

    f_name = Path('sim') / f'collected-tiles_evals-{evals}_target-score_{target_score}.json'
    if not f_name.is_file():
        run_simulation(f_name, evals, start_arr, target_score)
    tiles_collected = json_in(f_name)

    # Compute mean and SD over all runs
    n = len(tiles_collected)
    p = np.count_nonzero(tiles_collected) / n
    mean_score = np.mean([t for t in tiles_collected if t])
    se = np.sqrt(p * (1 - p) / n)  # Standard error
    z = norm.ppf(1 - alpha / 2)  # Z-score
    err = z * se
    is_success = p - err < expectation < p + err
    return expectation, p, err, is_success, mean_score


if __name__ == '__main__':
    # Print a data frame containing our simulation outcomes
    print(pd.DataFrame.from_dict(
        data={test: collect_simulation_results(test, evals=100_000, alpha=0.01) for test in Tiles.values},
        orient='index',
        columns=[
            'expected win rate',
            'simulated win rate',
            'SD',
            'success',
            'mean_tile',
        ],
    ).to_string(
        formatters={
            'expected win rate': '{:,.3%}'.format,
            'simulated win rate': '{:,.3%}'.format,
            'SD': '{:,.3%}'.format,
        }
    ))
