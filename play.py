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
min_tile = 21
max_tile = 36
# states = pickle_in('states.pkl')


def dict_sum(dict1: dict, dict2: dict) -> dict:
    # Adds the values of dict2 to dict1
    dict1.update({k: min(v + dict2[k], 1) for k, v in dict1.items() if k in dict2})
    dict1.update({k: v for k, v in dict2.items() if k not in dict1})
    return dict1


def dict_div(dict1: dict, denominator: int) -> dict:
    # Divides all values in a dictionary by a denominator
    return {k: v / denominator for k, v in dict1.items()}


def score_collection(col: np.ndarray) -> int:
    # Calculates the value of the collection, zero if no worms
    has_worms = col[-1] > 0
    score = (col * face_values).sum()
    return score if has_worms else 0


def remaining_pickups(collection: np.ndarray) -> np.ndarray:
    # Create an array of size [ N_ROLLS x N_DIE_FACES ]

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


def work_tree(collection: np.ndarray) -> dict:
    # Calculate the score of the current collection
    score = score_collection(collection)

    # Set the chance for all tiles with value >= the current score to 100%
    tile_chances = {t: 1.0 for t in range(min_tile, min(score + 1, max_tile + 1))}

    # Get the number of dice that can be rolled
    n_free_dice = n_dice - collection.sum()

    # Get the number of faces that can still be collected
    n_free_faces = len(face_values) - np.count_nonzero(collection)

    # Return if no die faces left to be collected OR if no dice left to roll
    if not n_free_faces or not n_free_dice:
        return tile_chances

    # List all possible outcomes of rolling the remaining dice
    outcomes = collection + remaining_pickups(collection)

    # Calculate the total number of possible outcomes
    n_outs = len(face_values) * n_free_dice

    # Divide the chance of each outcome by the total number of possible
    # outcomes. Sum all outcomes. dict_sum directly assigns to tile_chances.
    [dict_sum(tile_chances, dict_div(work_tree(o), n_outs)) for o in outcomes]
    return tile_chances


def roll_dice(collection: np.ndarray) -> np.ndarray:
    # Get the number of dice left to roll
    n_free_dice = n_dice - collection.sum()

    if not n_free_dice:
        raise ValueError('All dice have been collected. No dice left to roll.')

    # Generate a list of random dice faces outcomes
    r_ints = np.random.randint(1, len(face_values) + 1, size=n_free_dice)

    # Return the frequency of die face values
    return np.array([(r_ints == i).sum() for i in range(1, len(face_values) + 1)])


def pickup_results(collection: np.array, roll: np.array) -> pd.DataFrame:
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
        return None

    # Preallocate an array of all possible new collections
    new_coll = collection[np.newaxis, :].repeat(len(ops), 0)

    # Assign each picked up die to the possible new collection
    for ops_n, (idx, f_r) in enumerate(ops):
        new_coll[ops_n, idx] = f_r

    # Extract face values from the options
    faces = map(lambda x: x[0] + 1, ops)

    # Calculate the chance on outcomes of each pick up
    chances = (work_tree(n_c) for n_c in new_coll)

    # Concatenate the chances into one dataframe
    return pd.concat([pd.Series(
        data=chance,
        name=face,
        dtype=float) for chance, face in zip(chances, faces)],
        axis=1)


collection_state = np.zeros(len(face_values), dtype=int)
dice_roll = roll_dice(collection_state)
dice_roll = np.array((1, 1, 3, 1, 1, 1))
df = pickup_results(collection_state, dice_roll)
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
# fig.colorbar(h, label='Chance on score')
# fig.tight_layout()
# fig.show()

