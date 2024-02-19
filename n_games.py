import numpy as np
from utils import flatten
# number of dice = 8
# number of die faces = 6

n_combinations = 0
for collected_dice in range(1, 8 + 1):
    for collected_faces in range(1, 6 + 1):
        if collected_faces > collected_dice:
            continue
        required_dice = [6 - i for i in range(collected_faces)]
        rest_dice = [collected_faces] * (collected_dice - collected_faces)
        n_combinations += np.prod(required_dice + rest_dice)
print(f'The game has {n_combinations} possible outcomes')



c = np.sum(flatten([[np.prod([6 - i for i in range(f + 1)] + [f + 1] * (d - f)) for f in range(6) if f <= d ] for d in range(8)]))
print(f'The game has {c} possible outcomes')
