import numpy as np
# from utils import pickle_in
import pandas as pd

"""
Definitions:
- Collection: dice that have been picked up
- Free dice: the number of dice not in the collection
- Free face: a die face not present in the collection
"""

# Static
face_values = (1, 2, 3, 4, 5, 5)
n_dice = 8
tile_values = {
    21: 1, 22: 1, 23: 1, 24: 1, 25: 2, 26: 2, 27: 2, 28: 2,
    29: 3, 30: 3, 31: 3, 32: 3, 33: 4, 34: 4, 35: 4, 36: 4,
}



def dict_sum(dict1: dict, dict2: dict) -> dict:
    """
    Sum two dictionaries by adding values from dict2 to dict1, if present, and
    adding key-value pairs from dict2 to dict1 if the keys were not present.

    :param dict1: The first dictionary.
    :param dict2: The second dictionary.
    :return: A new dictionary with the combined key-value pairs.
    """
    # Adds the values of dict2 to dict1, if present in both
    dict1.update({k: min(v + dict2[k], 1) for k, v in dict1.items() if k in dict2})
    # Adds the keys of dict 2, along with values, to dict 1.
    dict1.update({k: v for k, v in dict2.items() if k not in dict1})
    return dict1


def dict_div(dict1: dict, denominator: int) -> dict:
    """
    Divide each value in the input dictionary by the specified denominator.

    :param dict1: The dictionary whose values are to be divided.
    :param denominator: The value to divide each dictionary value by.
    :return: A new dictionary with the values divided by the denominator.
    """
    # Divide each value by the denominator
    divided_dict = {k: v / denominator for k, v in dict1.items()}
    return divided_dict


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


def remaining_pickups(collection: np.ndarray) -> np.ndarray:
    """
    Determine all possible pickups of dice faces that have not been collected.

    :param collection: A numpy array representing the collection of dice faces.
    :return: A numpy array representing the possible pickups.
    """
    # Find face values that have not been collected yet
    free_faces = np.where(collection == 0)[0]

    # Get the total number of free dice
    n_free_dice = n_dice - collection.sum()

    pickups = []
    # For each free die face...
    for face in free_faces:
        # Create an empty template
        n_face = np.zeros([n_free_dice, len(face_values)], int)
        # Set the number that can be picked up, from 1 to n_free_dice
        n_face[:, face] = np.arange(1, n_free_dice + 1)
        # Add the result to the possible pickups
        pickups.append(n_face)

    # Return a concatenation of all free faces
    return np.concatenate(pickups)


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


def probability_tree(collection: np.ndarray) -> dict:
    """
    Recursively calculate the probabilities associated with reaching each tile
    value based on the current collection of dice faces.

    :param collection: A numpy array representing the collection of dice faces.
    :return: A dictionary mapping tile values to their probabilities.
    """
    # Calculate the score of the current collection
    score = count_score(collection)

    # Set the probability for all tiles with value >= the current score to 100%
    tile_probs = {t: 1.0 for t in tile_values if score >= t}

    # Get the number of dice that can be rolled
    n_free_dice = n_dice - collection.sum()

    # Get the number of faces that can still be collected
    n_free_faces = len(face_values) - np.count_nonzero(collection)

    # Return if no die faces left to be collected OR if no dice left to roll
    if not n_free_faces or not n_free_dice:
        return tile_probs

    # List all possible outcomes of rolling the remaining dice
    outcomes = collection + remaining_pickups(collection)

    # Calculate the total number of possible outcomes
    n_outs = len(face_values) * n_free_dice

    # Divide the probability of each outcome by the total number of possible
    # outcomes. Sum all outcomes. dict_sum directly assigns to tile_probs.
    [dict_sum(tile_probs, dict_div(probability_tree(o), n_outs)) for o in outcomes]
    return tile_probs


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
    probs = (probability_tree(n_c) for n_c in new_coll)

    # Concatenate the probs into one dataframe
    return pd.concat([pd.Series(
        data=prob,
        name=face,
        dtype=float) for prob, face in zip(probs, faces)],
        axis=1)


collection_state = np.zeros(len(face_values), dtype=int)
dice_roll = roll_dice(collection_state)
dice_roll = np.array((1, 1, 3, 1, 1, 1))
df = analyze_turn(collection_state, dice_roll)
print('You rolled:', dice_roll)
print(df.to_string(formatters={c: '{:,.0%}'.format for c in df.columns}))

# from matplotlib import pyplot as plt
# fig, ax = plt.subplots(figsize=(3, 5))
# h = ax.imshow(df, clim=[0, 0.5])
# ax.set_yticklabels(np.arange(min_tile, max_tile + 1))
# ax.set_xticks(np.arange(0, len(df.columns)))
# ax.set_xticklabels(df.columns)
# ax.set_title(dice_roll)
# ax.set_ylabel('Score')
# ax.set_xlabel('Die face')
# fig.colorbar(h, label='Probability of score')
# fig.tight_layout()
# fig.show()

