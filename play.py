from utils import pickle_in, pickle_out
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from collections import Counter

face_values = (1, 2, 3, 4, 5, 5)

n_dice = 8

tile_values = {
    21: 1, 22: 1, 23: 1, 24: 1, 25: 2, 26: 2, 27: 2, 28: 2,
    29: 3, 30: 3, 31: 3, 32: 3, 33: 4, 34: 4, 35: 4, 36: 4,
}

# Load the probabilities file
probs_file = Path('probabilities.pkl')
if not probs_file.is_file():
    pickle_out({}, probs_file)
probabilities = pickle_in(probs_file)


def pd_print(data: pd.DataFrame, transpose=False):
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    data.index.name = 'tile'
    if not transpose:
        data = pd.concat([data, pd.Series(data=tile_values, name='value', dtype=int)], axis=1)
        data = data.set_index(['value'], append=True)
        data = data.fillna(0)
    if transpose:
        data = data.T
    print(data.to_string(formatters={c: '{:,.1%}'.format for c in data.columns}))




def count_score(collection: np.ndarray) -> int:
    """
    Count the score based on an array of die face frequencies.

    :param collection: A numpy array representing the faces of the dice.
    :return: The calculated score based on the given rules.
    """
    # Check if there is at least one worm
    has_worms = collection[-1] > 0
    # Multiply the frequency of each die face with its value
    score = (collection * face_values).sum()
    return score if has_worms else 0


def roll_dice(collection: np.ndarray) -> np.ndarray:
    """
    Simulate rolling the remaining dice.
    Return the frequency of each rolled die face value.

    :param collection: A numpy array representing the collection of dice faces.
    :return: A numpy array of the frequency of each die face  after rolling.
    """
    # Get the number of dice left to roll
    n_free_dice = n_dice - collection.sum()

    if not n_free_dice:
        raise ValueError('All dice have been collected. No dice left to roll.')

    # Generate a list of random dice face outcomes
    r_ints = np.random.randint(1, len(face_values) + 1, size=n_free_dice)

    # Return the frequency of die face values
    return np.array([(r_ints == i).sum() for i in range(1, len(face_values) + 1)])


def get_best_probs(collection: np.array, pickups: np.array) -> pd.DataFrame:
    # if np.count_nonzero(pickups) > 4:
    #     xx
    # Calculate the chance of this pickup to occur.
    # For example (0, 0, 1, 1, 0, 0) is twice as likely to occur as (0, 0, 0, 2, 0, 0)

    # Create a new collection for each situation
    new_collections = np.repeat(collection[np.newaxis, :], np.count_nonzero(pickups), 0)

    # Assign the die that has been picked to the right location in the new collection
    for collection_idx, face_idx in enumerate(np.where(pickups != 0)[0]):
        new_collections[collection_idx, face_idx] = pickups[face_idx]

    # For each new collection in this situation, get the probability that a score can be achieved.
    all_roll_outcome = pd.DataFrame([get_score_probabilities(n_c) for n_c in new_collections])

    # Take the max because the % chance is the chance on an intended strategy to achieve a certain score
    best_roll_outcome = all_roll_outcome.max()
    return best_roll_outcome


def explore_dice_rolling(collection: np.ndarray, verbose=False) -> pd.Series:
    # Get the number of dice that can still be rolled
    n_free_dice = n_dice - collection.sum()

    # Retrieve all ways the dice can land and be picked up
    possible_rolls = list(product(range(1, len(face_values) + 1), repeat=n_free_dice))

    # Retrieve the best probabilities that can be achieved in all situations.
    possible_frequencies = [tuple([np.sum(np.array(r) == i) for i in range(1, len(face_values) + 1)]) for r in possible_rolls]
    # More specifically, a 'situation' is a unique outcome from rolling dice.
    best_prob_per_pickup = []

    collection_mask = collection == 0
    for pickups, p in Counter(possible_frequencies).items():
        # This is all ways the dice could land, minus the dice we cannot pick up
        valid_pickups = collection_mask * pickups
        # Get the best outcome for different target tiles
        best_roll_outcome = get_best_probs(collection, valid_pickups)
        # We add the outcome of this situation to a list of all situations
        best_prob_per_pickup.append(best_roll_outcome * p / len(possible_rolls))

    # Print probabilities of all underlying situations
    if verbose:
        print(f'Valid pick ups and outcomes for {collection}: ')
        # Map column names
        cols = {k: v for k, v in enumerate(
            [''.join(map(str, pickups)) + f' {p: .1%}' for pickups, p in Counter(possible_frequencies).items()])}
        # Concat array as normal
        df = pd.concat(best_prob_per_pickup, axis=1).fillna(0).rename(columns=cols)
        # Undo probability correction
        df /= Counter(possible_frequencies).values()  # sums to 1
        # Print underlying probabilities
        print(df.to_string(formatters={c: '{:,.1%}'.format for c in df.columns}))
        return df
    # Concat and sum the outcomes of all situations
    roll_probs = pd.concat(best_prob_per_pickup, axis=1).fillna(0).sum(1)
    # Update the probability after rolls with that based on the existing collection
    return roll_probs


def get_score_probabilities(collection: np.array, verbose=False) -> pd.Series:
    # Define serializable key to store and retrieve probabilities
    key = tuple(collection)
    if key in probabilities:
        return probabilities[key]

    # Get the number of dice that can still be rolled
    n_free_dice = n_dice - collection.sum()

    # Get the number of die faces not yet collected
    n_free_faces = (collection == 0).sum()

    # Get the score value of the current collection of dice
    score = count_score(collection)

    # Tiles with a value <= the score can be obtained with 100% certainty
    score_probs = pd.Series({t: 1.0 for t in tile_values if score >= t})

    # If we cannot roll or collect dice we are left with our current score
    if n_free_dice == 0 or n_free_faces == 0:
        probabilities[key] = score_probs
        return probabilities[key]

    # Get the probabilities of rolling dice
    roll_probs = explore_dice_rolling(collection, verbose)
    roll_probs.update(score_probs)
    score_probs = roll_probs

    # Set and return the value
    probabilities[key] = score_probs
    return probabilities[key]


def analyze_turn(collection: np.array, roll: np.array) -> pd.DataFrame:
    """
    Analyze the possible outcomes of picking up dice from the current collection
    based on the rolled dice faces.

    :param collection: A numpy array representing the current collection of dice faces.
    :param roll: A numpy array representing the rolled dice faces.
    :return: A pandas DataFrame containing the probabilities of each possible outcome.
    """
    if roll.sum() + collection.sum() < n_dice:
        d = 'few dice' if n_dice - roll.sum() + collection.sum() > 1 else 'die'
        raise ValueError(f'You are missing a {d}. Look under the table.')
    elif roll.sum() + collection.sum() > n_dice:
        raise ValueError('You are using too many dice!')

    # List the pickup options: each die face we can pick up with its frequency:
    # - idx: index
    # - f_r: frequency in roll (should be > 0 to be picked up)
    # - f_c: frequency in collection (should be 0 to be picked up)
    ops = [(idx, f_r) for idx, (f_r, f_c) in enumerate(zip(roll, collection)) if not f_c and f_r]
    if not any(ops):
        return pd.DataFrame(None)

    # Preallocate an array of all possible new collections
    new_coll = collection[np.newaxis, :].repeat(len(ops), 0)

    # Assign each picked up die to the possible new collection
    for ops_n, (idx, f_r) in enumerate(ops):
        new_coll[ops_n, idx] = f_r

    # Extract face values from the options
    faces = map(lambda x: x[0] + 1, ops)

    # Calculate the probability on outcomes of each pick up
    probs = (get_score_probabilities(n_c) for n_c in new_coll)

    # Concatenate the probs into one dataframe
    probs_df = pd.concat(
        objs=[pd.Series(data=tile_values, name='value', dtype=int)] + [pd.Series(
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


if not any(probabilities):
    print('Initializing probabilities file')
    for collected_dice in list(range(9))[::-1]:
        for c in product(range(n_dice + 1), repeat=len(face_values)):
            if np.sum(c) == collected_dice:
                if c not in probabilities:
                    get_score_probabilities(np.array(c))
        pickle_out(probabilities, probs_file)
        print('Done with', collected_dice)








