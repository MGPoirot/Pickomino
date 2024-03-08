import pickle
from pathlib import Path
import numpy as np
import json
import os

names = ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve', 'Frank', 'Grace']

players = []


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

    def pop(self, tile):
        if self[tile - min(self.values.keys())] == 0:
            raise ValueError('Cannot pop popped tile!')
        self[tile - min(self.values.keys())] = 0
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
        Calculate the number of free tiles remaining.
        :return: The number of free tiles remaining.
        """
        return int(self.sum())


    @property
    def free_tiles(self) -> bool:
        """
        Check if there are free tiles remaining.
        :return: True if there are free tiles remaining, False otherwise.
        """
        return bool(self.n_free_tiles)

    def gain(self, tile: int) -> None:
        self[tile - min(self.values.keys())] = 1

    def lose(self, tile: int) -> None:
        self[tile - min(self.values.keys())] = 0

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'Tiles:       ' + ' '.join([f'|{t}|' if k else '    ' for t, k in zip(self.values, self)]) + '\n' \
               '             ' + ' '.join([f'| {s}|' if k else ' __ ' for (_, s), k in zip(self.values.items(), self)])


class Player(list):
    def __init__(self,
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
            raise ValueError(f'A player possesses more tiles than available in the game')

        self.name = names.pop(0) if name is None else name
        players.append(self)

        self.params = {}
        if params is not None:
            self.params = params
            for kv in params.items():
                self.__setattr__(*kv)

    def reset(self):
        self.clear()

    @property
    def score(self) -> int:
        return int(np.sum([Tiles.values[t] for t in self]))

    @property
    def position(self) -> int:
        return self.score - np.max([p.score for p in players if p is not self])

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{self.name} {" ".join([f"[{t}]" for t in self])}'


def json_in(source: Path | str):
    with open(source, 'r') as file:
        return json.load(file)


def json_out(obj: object, target: Path | str):
    with open(target, 'w') as file:
        json.dump(obj, file, indent=4, sort_keys=True)

