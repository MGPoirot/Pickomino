from utils import json_in, json_out
from evaluate_turn import present_roll_options, get_turn_probabilities, Dice
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

# What is the value of a tile face?
# 1 worm can be got 89.3% of turns
# If you are able to get 2 worms, the only way a competitor can catch up is
# - A pair of turns exist where they get a worm and you not (9.6%)
# - They get two worms in a turn (68.0%)

"""
WE NEED TO ASSIGN VALUES TO TILES
BASED ON THE CURRENT STATE OF THE GAME, 
WHAT DOES THE ACQUISITION OF A TILE ADD TO THE COMPETITIVE POSITION OF THE PLAYER?
"""


def run_simulation(risk, evals=10_000):
    min_scaled_chance = 1 - risk

    start_arr = np.array((0, 0, 0, 0, 0, 0)).view(Dice)

    f_name = Path('sim') / f'collected-tiles_evals-{evals}_target-scaledrisk_{risk:.2f}.json'
    if f_name.is_file():
        tiles_collected = json_in(f_name)
    else:
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

                values = [5 - v for v in Tiles.values]
                scaled_chance = chance_chart.pow(values, axis=0)

                mask = ((scaled_chance > min_scaled_chance).sum(1) > 0)
                if any(mask):
                    idx = int(chance_chart[mask].iloc[-1].idxmax()[0]) - 1
                else:
                    # We are beggars and cant choose an option that suits our risk profile
                    # We assume we have go for reaching the 21 tile and shoot our best shot
                    idx = int((chance_chart - min_scaled_chance).iloc[0].idxmax()[0]) - 1

                # Insert the dice into the dice
                collection[idx] = roll[idx]

                forced_exit = not collection.free_dice or not collection.free_faces

                # Evaluate
                chance_chart = get_turn_probabilities(collection)
                roll_chance = chance_chart[get_turn_probabilities(collection) != 1]
                score = collection.score
                score = score if score >= np.min(tuple(Tiles.values)) else 0
                voluntary_exit = roll_chance.max() <= chance_taken and score > 0

                # 3) Exit the game for one of three reasons
                if forced_exit or voluntary_exit:
                    tiles_collected.append(score)
                    break
        json_out(tiles_collected, f_name)

    # Compute mean and SD over all runs
    win_rate = np.count_nonzero(tiles_collected) / evals
    mean = np.mean(tiles_collected)
    mean_s = np.mean([t for t in tiles_collected if t])
    median = np.median(tiles_collected)
    median_s = np.median([t for t in tiles_collected if t])
    return win_rate, mean, mean_s, median, median_s


if __name__ == '__main__':
    # Define simulation scenarios to test
    risks_taken = np.linspace(0, 1, 51)

    # Create a data frame to store our simulation outcomes
    simulation_outcomes = pd.DataFrame(columns=[
        'risk_taken',
        'win_rate',
        'mean_score',
        'mean_score_of_successes',
        'median_score',
        'median_score_of_successes',
    ]).set_index(['risk_taken'])

    # Perform the simulations
    for risk_taken in risks_taken:
        simulation_outcomes.loc[risk_taken] = run_simulation(risk_taken, evals=100)

    # Format and print the simulation results
    formatters = {
        'win_rate': '{:,.1%}'.format,
    }
    print(simulation_outcomes.to_string(formatters=formatters))
