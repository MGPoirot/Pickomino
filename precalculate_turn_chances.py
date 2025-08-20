from utils import pickle_in, pickle_out, Dice, Tiles, json_out, json_in
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from collections import Counter


def get_roll_probabilities(dice: Dice, pickups: np.array) -> pd.DataFrame:
    """
    Compute the *best achievable* tile-win probabilities for a single roll outcome.

    Given the current `dice` state and a vector `pickups` (length = n_faces) indicating how many dice
    of each face were rolled (and thus are eligible to be picked), this function enumerates all legal
    single-face pickup choices and evaluates the downstream turn probabilities for each resulting state.
    It returns, per tile, the maximum probability across those choices (i.e., optimal play).

    Parameters
    ----------
    dice : Dice
        The current dice collection/state. Must be numpy-like and indexable per face.
    pickups : np.array
        Length-n_faces array of counts indicating how many dice of each face are available to pick.

    Returns
    -------
    pd.Series
        A Series indexed by tile value giving, for each tile, the highest probability of being able
        to claim it after making the best pickup choice for this roll outcome.

    Notes
    -----
    - Internally, this calls `get_turn_probabilities` on each candidate post-pickup state, stacks
      the resulting Series into a DataFrame, and takes the column-wise max to keep the optimal choice.
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
    Aggregate tile-win probabilities over all possible outcomes of rolling the free dice.

    For the current state, enumerate all permutations of rolling `dice.n_free_dice` dice with
    `dice.n_faces` sides. Group permutations into frequency histograms to avoid recomputation,
    evaluate the best post-pickup outcome for each histogram, weight by multiplicity, and sum.

    Parameters
    ----------
    dice : Dice
        The current dice collection/state.

    Returns
    -------
    pd.Series
        A Series indexed by tile value with probabilities for achieving each tile *after* rolling,
        assuming optimal pickup strategy from the roll outcome onward.

    Implementation details
    ----------------------
    - `product(range(1, n_faces+1), repeat=n_free_dice)` enumerates permutations.
    - A histogram of face counts (tuple of length n_faces) represents an equivalence class.
    - Multiplicity is tracked by `Counter`; contributions are weighted by count / total permutations.
    - Illegal pickups (faces already collected) are masked out by `(dice == 0)`.
    """
    # Retrieve all posisble ways the dice can land and be picked up
    possible_rolls = list(product(range(1, dice.n_faces + 1), repeat=dice.n_free_dice))

    # Retrieve the best probabilities that can be achieved in all situations.
    possible_frequencies = [tuple([np.sum(np.array(r) == i) for i in range(1, dice.n_faces + 1)]) for r in possible_rolls]
    # More specifically, a 'situation' is a unique outcome from rolling dice.
    best_prob_per_pickup = []

    collection_mask = dice == 0
    for pickups, p in Counter(possible_frequencies).items():
        # This is all ways the dice could land, minus the dice we cannot pick up
        valid_pickups = collection_mask * pickups
        # Get the best outcome for different target tiles_p
        best_roll_outcome = get_roll_probabilities(dice, valid_pickups)
        # We add the outcome of this situation to a list of all situations
        best_prob_per_pickup.append(best_roll_outcome * p / len(possible_rolls))

    # Concat and sum the outcomes of all situations
    roll_probs = pd.concat(best_prob_per_pickup, axis=1).fillna(0).sum(1)
    # Update the probability after rolls with that based on the existing dice
    return roll_probs


def get_turn_probabilities(dice: Dice) -> pd.Series:
    """
    Return optimal tile-win probabilities from the given `dice` state for the current rule.

    The result is memoized under `probabilities[dice.key]`. If the state has no legal continuation
    (no free dice to roll or no free faces to pick), the probability mass collapses to tiles whose
    value satisfies the current rule `compare(dice.score, tile)`.

    Parameters
    ----------
    dice : Dice
        The current dice collection/state.

    Returns
    -------
    pd.Series
        Series mapping tile value -> probability of being able to claim the tile by the end of the turn.

    Relies on
    ---------
    probabilities : Dict[hashable, pd.Series]
        Module-global memoization cache.
    compare : Callable[[int, int], bool]
        Module-global rule predicate: compare(current_score, tile_value) -> bool.
    """
    # Define serializable key to store and retrieve probabilities
    if dice.key in probabilities:
        return probabilities[dice.key]

    # Tiles with a value <= the score can be obtained with 100% certainty
    score_probs = pd.Series({t: 1.0 for t in Tiles.values if compare(dice.score, t)})

    # If we cannot roll or collect dice we are left with our current score
    if not dice.free_dice or not dice.free_faces:
        probabilities[dice.key] = score_probs
        return probabilities[dice.key]

    # Get the probabilities of rolling dice
    roll_probs = get_rolling_probabilities(dice)
    if any(score_probs):
        score_probs = score_probs.combine_first(roll_probs)
    else:
        score_probs = roll_probs

    # Set and return the value
    probabilities[dice.key] = score_probs
    return probabilities[dice.key]


def present_roll_options(dice: Dice, roll: np.array) -> pd.DataFrame:
    """
    Given a concrete roll (as face frequencies), evaluate outcomes for each legal single-face pickup.

    This is a helper for UI/debugging: for a specific roll, it shows the probability of eventually
    claiming each tile if you pick each eligible face.

    Parameters
    ----------
    dice : Dice
        Current state before picking from this roll.
    roll : np.array
        Length-n_faces vector with the number of dice rolled for each face.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by (tile, value). Each column corresponds to a pickup option named
        like "face (count)". Cell values are probabilities in [0, 1].

    Raises
    ------
    ValueError
        If `roll` is inconsistent with `dice` (too few or too many dice).
    """
    if roll.sum() + dice.sum() < dice.n_dice:
        d = 'few dice' if dice.n_dice - roll.sum() + dice.sum() > 1 else 'die'
        raise ValueError(f'You are missing a {d}. Look under the table.')
    elif roll.sum() + dice.sum() > dice.n_dice:
        raise ValueError('You are using too many dice!')

    # List the pickup options: each die face we can pick up with its frequency:
    # - tile_idx: index
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


def initialize() -> None:
    """
    Seed the memoization table by exploring from the default starting state `Dice()`.
    Safe to call multiple times; it reuses the global `probabilities` cache.
    """
    get_turn_probabilities(Dice())


tasks = {
    'minimal': lambda x, y: x >= y,
    'exactly': lambda x, y: x == y,
}

if __name__ == '__main__':
    """
    Compute and export precomputed probabilities for each rule in `tasks`.

    For each task:
      1) Load (or create) the memo cache file at precalculated_chances/<task>.pkl
      2) Set module-globals `probabilities` (dict) and `compare` (callable) for this run
      3) If the cache is empty, initialize from the starting state
      4) Persist the pickle, and emit a compact JSON with percentages

    Files written
    -------------
    - precalculated_chances/<task>.pkl : full memoization dictionary
    - <task>.json : {"STATEKEY": {"tile": percent, ...}, ...}
    """
    for task_name, compare in tasks.items():
        # Load the probabilities file
        probs_file = Path(f'precalculated_chances/{task_name}.pkl')
        if not probs_file.is_file():
            pickle_out({}, probs_file)
        probabilities = pickle_in(probs_file)

        if not any(probabilities):
            print(f'Initializing {task_name} file')
            initialize()
        pickle_out(probabilities, probs_file)

        json_probabilities = {}
        for state, v in probabilities.items():
            # Empty probability values are redundant
            if not len(v):
                continue
            # Round the probability values to whole percentages
            json_probabilities[''.join(map(str, state))] = {k: round(vv * 100, 1) for k, vv in v.items()}
        json_out(json_probabilities, f'precalculated_chances/{task_name}.json')