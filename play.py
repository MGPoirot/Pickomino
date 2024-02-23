import numpy as np
from utils import pickle_in
import pandas as pd


# Static
face_values = (1, 2, 3, 4, 5, 5)
states = pickle_in('states.pkl')


def dict_sum(dict1: dict, dict2: dict) -> dict:
    dict1.update({k: min(v + dict2[k], 1) for k, v in dict1.items() if k in dict2})
    dict1.update({k: v for k, v in dict2.items() if k not in dict1})
    return dict1


def dict_div(dict1: dict, denominator: int) -> dict:
    return {k: v / denominator for k, v in dict1.items()}


def sum_collection(col: np.ndarray) -> int:
    # Calculates the value of the collection, zero if no worms
    score = (col * face_values).sum()
    return score if bool(col[-1]) > 0 else 0


def get_valid_pickups(col: np.ndarray) -> np.ndarray:
    # Create an array of size [ N_ROLLS x N_DIE_FACES ]
    collectable_faces = np.where(col == 0)[0]
    n_collectable_dice = 8 - col.sum()
    all_faces = []
    for face in collectable_faces:
        n_face = np.zeros([n_collectable_dice, 6], int)
        n_face[:, face] = np.arange(1, n_collectable_dice + 1)
        all_faces.append(n_face)
    return np.concatenate(all_faces)


def work_tree(col: np.ndarray):
    collection_value = sum_collection(col)
    tile_chances = {t: 1.0 for t in range(21, min(collection_value + 1, 37))}

    # Return if there are no die faces left to be collected OR if no dice left to roll
    if np.count_nonzero(col) == len(face_values) or np.sum(col) == 8:
        return tile_chances

    n_collectable_dice = 8 - col.sum()

    # List all possible outcomes of rolling the remaining dice
    outcomes = col + get_valid_pickups(col)
    [dict_sum(tile_chances, dict_div(work_tree(o), 6 * n_collectable_dice)) for o in outcomes]
    return tile_chances


def roll_dice(col) -> tuple:
    random_numbers = np.random.randint(1, 7, size=8 - np.sum(col))
    return tuple([(random_numbers == i).sum() for i in range(1, len(face_values) + 1)])


def pickup_results(current_collection, roll):
    current_collection = np.array(current_collection)
    pickup_options = [(face_no, n_rolled) for face_no, (n_rolled, in_collection) in enumerate(zip(roll, collection)) if not in_collection and n_rolled]
    if not any(pickup_options):
        return None
    new_collections = current_collection[np.newaxis, :].repeat(len(pickup_options), 0)
    for c, (f, i) in enumerate(pickup_options):
        new_collections[c, f] = i
    return pd.concat([pd.Series(work_tree(new_collection), name=f+1, dtype=float) for new_collection, (f, _) in zip(new_collections, pickup_options)], axis=1)


collection = (0, 0, 0, 0, 0, 0)
dice_roll = roll_dice(collection)
dice_roll = (0, 0, 0, 0, 5, 5)
df = pickup_results(collection, dice_roll)
print('You rolled:', dice_roll)
print(df.to_string(formatters={c: '{:,.0%}'.format for c in df.columns}))

