import pickle
from pathlib import Path
import numpy as np
import json
import os
from typing import List


def flatten(lst: list) -> list:
    """
    Flattens a nested list that contains sub-lists.

    Args:
        :param lst: A nested list.
        :type lst: list

    Returns:
        :return: An unnested list.
        :rtype: list

    Example:
        > nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        > flatten(nested_list)
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    return [item for sublist in lst for item in sublist]


def pickle_in(source: str | Path):
    with open(source, 'rb') as file:
        return pickle.load(file)


def pickle_out(obj: dict, target: str | Path) -> None:
    with open(target, 'wb') as file:
        pickle.dump(obj, file)


def mk_tmp(target: str | Path, exist_ok=False):
    if os.path.isfile(target) and not exist_ok:
        raise FileExistsError(f'File exists at "{target}"')
    with open(target, 'w') as _:
        pass


class Dice(np.ndarray):
    """
    Represents a collection of die face frequencies.
    An example for eight dice with six faces is: (0, 0, 3, 4, 0, 1)
    This example means that three threes, four fours and one worm die face have been collected.
    """

    values = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'w': 5}

    n_faces = len(values)

    n_dice = 8

    def __new__(cls, state: tuple | list | np.ndarray | None = None):
        if state is None:
            state = np.zeros(cls.n_faces, dtype=int)
        elif isinstance(state, tuple | list):
            state = np.array(state)
        elif isinstance(state, np.ndarray):
            pass
        else:
            raise TypeError('Dice state is not of type tuple | list | np.ndarray')

        if not len(state) == cls.n_faces:
            raise ValueError(f'Dice state is not of length {cls.n_faces}')

        return state.view(Dice)

    @property
    def key(self):
        """
        Get a hashable key that uniquely identifies the dice.
        :return: A tuple representing the dice.
        """
        return tuple(self)

    @property
    def n_free_dice(self) -> int:
        """
        Calculate the number of free dice remaining.
        :return: The number of free dice remaining.
        """
        return int(self.n_dice - self.sum())

    @property
    def free_dice(self) -> bool:
        """
        Check if there are free dice remaining.
        :return: True if there are free dice remaining, False otherwise.
        """
        return self.n_free_dice > 0

    @property
    def n_free_faces(self) -> int:
        """
        Calculate the number of free faces remaining.
        :return The number of free faces remaining as integer.
        """
        return int((self == 0).sum())

    @property
    def free_faces(self) -> bool:
        """
        Check if there are free faces remaining.
        :return: True if there are free faces remaining, False otherwise.
        """
        return self.n_free_faces > 0

    @property
    def score(self) -> int:
        """
        Count the score based on an array of die face frequencies.
        :return: The calculated score based on the given rules.
        """
        # Check if there is at least one worm
        has_worms = self[-1] > 0
        # Multiply the frequency of each die face with its value
        score = (self * tuple(self.values.values())).sum()
        return int(score if has_worms else 0)

    def roll(self) -> np.ndarray:
        """
        Simulate one random roll of all dice that have not been collected.
        Return the frequency of each rolled die face value.

        :param dice: A Dice object representing the dice of die face frequencies.
        :return: A numpy array representing the frequency of each die face after rolling.
        """
        if not self.free_dice:
            raise ValueError('All dice have been collected. No dice left to roll.')

        # Generate a list of random dice face outcomes
        r_ints = np.random.randint(1, self.n_faces + 1, size=self.n_free_dice)

        # Return the frequency of die face values
        return np.array([(r_ints == i).sum() for i in range(1, self.n_faces + 1)])


class Tiles(np.ndarray):
    """
    Represents a collection of die face frequencies.
    An example for eight dice with six faces is: (0, 0, 3, 4, 0, 1)
    This example means that three threes, four fours and one worm die face have been collected.
    """

    values = {
        21: 1, 22: 1, 23: 1, 24: 1, 25: 2, 26: 2, 27: 2, 28: 2,
        29: 3, 30: 3, 31: 3, 32: 3, 33: 4, 34: 4, 35: 4, 36: 4,
    }

    n_tiles = len(values)

    def __contains__(self, tile: int):
        return bool(self[tile - min(self.values.keys())])

    def __new__(cls, state: tuple | list | np.ndarray | None = None):
        if state is None:
            state = np.ones(cls.n_tiles, dtype=int)
        elif isinstance(state, tuple | list):
            state = np.array(state)
        elif isinstance(state, np.ndarray):
            pass
        else:
            raise TypeError('Dice state is not of type tuple | list | np.ndarray')

        if not len(state) == cls.n_tiles:
            raise ValueError(f'Tile state is not of length {cls.n_tiles}')

        return state.view(Tiles)

    def is_available(self, tile: int):
        return self[tile - min(self.values.keys())]

    def transfer(self, tile: int | str):
        if isinstance(tile, str):
            tile = int(tile[1:])
        if self[tile - min(self.values.keys())] == 0:
            raise ValueError('Cannot transfer missing tile!')
        self.lose(tile)
        return tile

    @property
    def key(self):
        """
        Get a hashable key that uniquely identifies the dice.
        :return: A tuple representing the dice.
        """
        return tuple(self)

    @property
    def n_free_tiles(self) -> int:
        """
        Calculate the number of free tiles_p remaining.
        :return: The number of free tiles_p remaining.
        """
        return int(self.sum())

    @property
    def free_tiles(self) -> bool:
        """
        Check if there are free tiles_p remaining.
        :return: True if there are free tiles_p remaining, False otherwise.
        """
        return bool(self.n_free_tiles)

    def gain(self, tile: int | str) -> None:
        if isinstance(tile, str):
            tile = int(tile[1:])
        self[tile - min(self.values.keys())] = 1

    def lose(self, tile: int | str) -> None:
        if isinstance(tile, str):
            tile = int(tile[1:])
        self[tile - min(self.values.keys())] = 0

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Tiles:           ' + ' '.join([f'|{t}|' if k else '    ' for t, k in zip(self.values, self)]) + '\n' \
               '                 ' + ' '.join([f'| {s}|' if k else ' __ ' for (_, s), k in zip(self.values.items(), self)])


def log(key, *values, i=1):
    if any(key):
        sep = ':'
    else:
        sep = ' '
    print(f'{" " * 4 * i}{key}{sep}'.ljust(16), *values)


class Players(list):
    def __init__(self, params: List[dict] | None = None, n: int | None = None):
        super().__init__()
        if params is None:
            params = [{}] * n
        elif n is not None:
            raise ValueError('Argument "n" does nothing when a list of params is given.')

        self.names = []
        for param in params:
            state = None if 'state' not in param else param.pop('state')
            name = self.name() if 'name' not in param else param.pop('name').capitalize()
            self.append(Player(self, state, name, params=param))

    def name(self, index=-1):
        if len(self.names) == 0:
            self.names.extend(['Grace', 'Frank', 'Eve', 'Dave', 'Charlie', 'Bob', 'Alice'])
        return self.names.pop(index)


class Player(list):
    def __init__(self,
                 parent,
                 state: tuple | list | np.ndarray | None = None,
                 name: str | None = None,
                 params: dict | None = None):
        super().__init__()

        if state is None:
            state = []
        elif isinstance(state, tuple | np.ndarray):
            state = list(state)
        else:
            raise TypeError('Player state is not of type tuple | list | np.ndarray')

        if len(state) > Tiles.n_tiles:
            raise ValueError(f'A player possesses more tiles_p than available in the game')

        self.parent = parent
        self.name = self.parent.name() if name is None else name

        self.params = {}
        if params is not None:
            self.params = params
            for kv in params.items():
                self.__setattr__(*kv)

    def __call__(self, param, fallback):
        if hasattr(self, param):
            return getattr(self, param)
        else:
            return fallback

    def transfer(self, tile: int | str):
        if isinstance(tile, str):
            tile = int(tile[1:])
        return self.pop(self.index(tile))

    def reset(self):
        self.clear()

    @property
    def score(self) -> int:
        return int(np.sum([Tiles.values[t] for t in self]))

    @property
    def position(self) -> int:
        return self.score - np.max([p.score for p in self.parent if p is not self])

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{self.name} {self.params} {" ".join([f"[{t}]" for t in self])}'


def json_in(source: Path | str):
    with open(source, 'r') as file:
        return json.load(file)


def json_out(obj: object, target: Path | str):
    with open(target, 'w') as file:
        json.dump(obj, file, indent=4, sort_keys=True)

