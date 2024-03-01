from utils import pickle_in, pickle_out, Dice, Tiles
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from collections import Counter


def get_roll_probabilities(dice: Dice, pickups: np.array) -> pd.DataFrame:
    """
    Calculate the probabilities of achieving a score based for a die roll outcome.

    :param dice: A Dice object representing the current dice of dice faces.
    :param pickups: A numpy array representing the pickup options for each face.
    :return: A pandas DataFrame containing the probabilities of achieving each score for each pickup scenario.
    """
    # Create a new dice for each situation
    new_collections = np.repeat(dice[np.newaxis, :], np.count_nonzero(pickups), 0)
    for collection_idx, face_idx in enumerate(np.where(pickups != 0)[0]):
        # Assign the die that can be picked to its location in the new dice
        new_collections[collection_idx, face_idx] = pickups[face_idx]

    # For each new dice in this situation, get the probability that a score can be achieved.
    all_roll_outcome = pd.DataFrame([get_turn_probabilities(n_c) for n_c in new_collections])

    # Take the max because the % chance is the chance on an intended strategy to achieve a certain score
    best_roll_outcome = all_roll_outcome.max()
    return best_roll_outcome


def get_rolling_probabilities(dice: Dice) -> pd.Series:
    """
    Calculate the probabilities of achieving tile scores for the rolling of dice.
    The rolling of dice includes all possible outcomes from rolling the dice that have not been collected.

    :param dice:
    :return:
    """
    # Retrieve all ways the dice can land and be picked up
    possible_rolls = list(product(range(1, dice.n_faces + 1), repeat=dice.n_free_dice))

    # Retrieve the best probabilities that can be achieved in all situations.
    possible_frequencies = [tuple([np.sum(np.array(r) == i) for i in range(1, dice.n_faces + 1)]) for r in possible_rolls]
    # More specifically, a 'situation' is a unique outcome from rolling dice.
    best_prob_per_pickup = []

    collection_mask = dice == 0
    for pickups, p in Counter(possible_frequencies).items():
        # This is all ways the dice could land, minus the dice we cannot pick up
        valid_pickups = collection_mask * pickups
        # Get the best outcome for different target tiles
        best_roll_outcome = get_roll_probabilities(dice, valid_pickups)
        # We add the outcome of this situation to a list of all situations
        best_prob_per_pickup.append(best_roll_outcome * p / len(possible_rolls))

    # Concat and sum the outcomes of all situations
    roll_probs = pd.concat(best_prob_per_pickup, axis=1).fillna(0).sum(1)
    # Update the probability after rolls with that based on the existing dice
    return roll_probs


def get_turn_probabilities(dice: Dice) -> pd.Series:
    """
    Calculate the probabilities of achieving tile scores for a turn.
    The turn includes the option to quit or continue rolling dice.

    :param dice:
    :return:
    """
    # Define serializable key to store and retrieve probabilities
    if dice.key in probabilities:
        return probabilities[dice.key]

    # Tiles with a value <= the score can be obtained with 100% certainty
    score_probs = pd.Series({t: 1.0 for t in Tiles.values if dice.score >= t})

    # If we cannot roll or collect dice we are left with our current score
    if not dice.free_dice or not dice.free_faces:
        probabilities[dice.key] = score_probs
        return probabilities[dice.key]

    # Get the probabilities of rolling dice
    roll_probs = get_rolling_probabilities(dice)
    roll_probs.update(score_probs)
    score_probs = roll_probs

    # Set and return the value
    probabilities[dice.key] = score_probs
    return probabilities[dice.key]


def present_roll_options(dice: Dice, roll: np.array) -> pd.DataFrame:
    """
    Analyze the possible outcomes of picking up dice from the current dice
    based on the rolled dice faces.

    :param dice: A numpy array representing the current dice of dice faces.
    :param roll: A numpy array representing the rolled dice faces.
    :return: A pandas DataFrame containing the probabilities of each possible outcome.
    """
    if roll.sum() + dice.sum() < dice.n_dice:
        d = 'few dice' if dice.n_dice - roll.sum() + dice.sum() > 1 else 'die'
        raise ValueError(f'You are missing a {d}. Look under the table.')
    elif roll.sum() + dice.sum() > dice.n_dice:
        raise ValueError('You are using too many dice!')

    # List the pickup options: each die face we can pick up with its frequency:
    # - idx: index
    # - f_r: frequency in roll (should be > 0 to be picked up)
    # - f_c: frequency in dice (should be 0 to be picked up)
    ops = [(idx, f_r) for idx, (f_r, f_c) in enumerate(zip(roll, dice)) if not f_c and f_r]
    if not any(ops):
        return pd.DataFrame(None)

    # Preallocate an array of all possible new collections
    new_coll = dice[np.newaxis, :].repeat(len(ops), 0)

    # Assign each picked up die to the possible new dice
    for ops_n, (idx, f_r) in enumerate(ops):
        new_coll[ops_n, idx] = f_r

    # Extract face values from the options
    faces = map(lambda x: f'{x[0] + 1} ({x[1]})', ops)

    # Calculate the probability on outcomes of each pick up
    probs = (get_turn_probabilities(n_c) for n_c in new_coll)

    # Concatenate the probs into one dataframe
    probs_df = pd.concat(
        objs=[pd.Series(data=Tiles.values, name='value', dtype=int)] + [pd.Series(
            data=prob,
            name=face,
            dtype=float
        ) for prob, face in zip(probs, faces)]
        , axis=1,
    ).fillna(0)

    # Give an existing index a name
    probs_df.index.name = 'tile'

    # Use the tile value as an index
    probs_df = probs_df.set_index(['value'], append=True)
    return probs_df


def initialize():
    # Calculate all probabilities and save
    get_turn_probabilities(Dice())
    pickle_out(probabilities, probs_file)


# Load the probabilities file
probs_file = Path('probabilities.pkl')
if not probs_file.is_file():
    pickle_out({}, probs_file)
probabilities = pickle_in(probs_file)


if __name__ == '__main__':
    if not any(probabilities):
        print('Initializing probabilities file')
        initialize()

