# # import numpy as np
# # from utils import flatten
# # # number of dice = 8
# # # number of die faces = 6
# #
# # n_combinations = 0
# # for collected_dice in range(1, 8 + 1):
# #     for collected_faces in range(1, 6 + 1):
# #         if collected_faces > collected_dice:
# #             continue
# #         required_dice = [6 - i for i in range(collected_faces)]
# #         rest_dice = [collected_faces] * (collected_dice - collected_faces)
# #         n_combinations += np.prod(required_dice + rest_dice)
# # print(f'The game has {n_combinations} possible outcomes')
# #
# #
# #
# # c = np.sum(flatten([[np.prod([6 - i for i in range(f + 1)] + [f + 1] * (d - f)) for f in range(6) if f <= d ] for d in range(8)]))
# # print(f'The game has {c} possible outcomes')
#
# from utils import flatten
#
# import numpy as np
# n_faces = 6
# n_dice = 8
# face_values = [1, 2, 3, 4, 5, 5]
# # collection = np.empty(F, dtype=int)
# # roll = np.empty(F, dtype=int)
#
# collection = np.array([0, 0, 0, 1, 2, 2])
# roll = np.array([1, 0, 1, 0, 0, 1])
#
# solutions = {}
#
# options_template = np.zeros((len(roll), len(collection)), dtype=int)
# options = options_template.copy()
#
#
# [np.put(options, face + face * len(collection), n) for face, n in enumerate(roll)]
#
# print(options)
# # Rules
# # collection.sum() + roll.sum() = 8
#
#
# def dice_mutations(collection: np.ndarray, n_faces=6, n_dice=8) -> list:
#     # Check how many dice we can roll
#     free_dice = n_dice - collection.sum()
#
#     # Check which faces we can still collect
#     free_faces = np.where(collection == 0)[0]
#
#     # If either prevents continuation of the game, return the current score
#     if not any(free_faces) or not free_dice:
#         return np.multiply(collection, face_values).sum()
#
#     # Create a dict to store all possible outcomes
#     outcomes = {i: 0 for i in [0] + list(range(21, 37))}
#
#     # Store the number of rolls that will result in a loss
#     outcomes[0] = (n_faces - len(free_faces)) * n_faces
#
#     # Go over all possible outcomes
#     for target_face in free_faces:
#         for n in range(1, free_dice + 1):
#             new_collection = collection.copy()
#             new_collection[target_face] = n
#             score = dice_mutations(new_collection)
#             if isinstance(score, int):
#                 if score < 21:
#                     outcomes[0] += 1
#                 else:
#                     outcomes[score] += 1
#             else:
#                 outcomes[score - 20] += 1
#     # outcomes[:, 1] /= n_rolls_total
#     return outcomes
#
#
# foo = dice_mutations(collection)
# print(foo)
#
# def give_options(col: np.ndarray, rol: np.ndarray) -> list:
#     # Calculate the number of dice that can still be thrown
#
#
#
#     # Calculate the outcome of an exit
#     x = (np.multiply(collection, face_values).sum(), 1.0)
#
#     # List which faces are allowed options
#     options = np.logical_and(collection == 0, roll > 0)
#     if not any(options):
#         return [x]
#
#     # Preallocate new collections that we
#     new_collections = collection[np.newaxis, :].repeat(options.sum(), axis=0)
#     for option_n, (f, n) in enumerate((f, n) for f, n in enumerate(options * roll) if n):
#         # Assign the number N of the corresponding face F
#         new_collections[option_n, f] = n
#
#
#
#

#########
import numpy as np
from itertools import product
from tqdm import tqdm
from utils import pickle_out, pickle_in
from pathlib import Path


def cartesian_product():
    return product(range(n_dice + 1), repeat=n_faces)


n_faces = 6
n_dice = 8
f_name = Path('states.pkl')

if not f_name.is_file():
    pickle_out({}, f_name)
states = pickle_in(f_name)

collections = [c for c in cartesian_product() if sum(c) <= n_dice and c not in states]

for collection in tqdm(collections):
    throws = np.array([c for c in cartesian_product() if sum(c) == n_dice - np.sum(collection)])
    states[collection] = throws
    pickle_out(states, f_name)

######


