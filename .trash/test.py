from scipy.stats import norm
from dice import roll_dice, present_roll_options, count_score, n_dice, get_turn_probabilities, get_roll_probabilities
import numpy as np
from utils import pickle_in, pickle_out
import numpy as np
import pandas as pd
import math
from pathlib import Path

states = pickle_in('states_valid.pkl')
probabilities = pickle_in(Path('probabilities.pkl'))

collection = np.array((0, 0, 0, 1, 0, 0))

rolls = states[tuple(collection)]

for pickups in rolls:
    if np.count_nonzero(pickups) > 4:
        xx
    get_roll_probabilities(collection, pickups)
