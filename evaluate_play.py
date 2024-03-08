import matplotlib.pyplot as plt

from utils import Tiles, Dice, Player, pickle_in
import numpy as np
import pandas as pd

probabilities = pickle_in('minimal.pkl')
steal = pickle_in('exactly.pkl')
n_competitors = 7
n_games = 10000
do_steal = True
players = [Player() for _ in range(n_competitors)]

# players = [Player(name='You')] + players


def break_turn():
    turn_types[game_n].append(0)
    if any(player):
        if any(tiles):
            # What is the rule when returning a tile, and nothing is there?
            last_tile = [v for k, v in zip(tiles, tiles.values) if k][-1]
            tiles.lose(last_tile)
        tile = player.pop()
        tiles.gain(tile)
        lost = tiles.values[tile]
        print(f'   LOST:  |{tile}|')
        print(f'          | {lost}|')
    else:
        print(f'   LOST')

turn_types = {}

win_counter = {p.name: 0 for p in players}
for game_n in range(n_games):
    tiles = Tiles()
    turn_types[game_n] = []
    [player.reset() for player in players]
    while any(tiles):
        for player in players:
            potential_loss = 0 if not any(player) else player[-1]

            steal_multiplier = lambda _: 2
            if hasattr(player, 'steal_reward'):
                if player.steal_reward == 'leader':
                    steal_multiplier = lambda victim: 2 if victim.position > 0 else 1
                elif player.steal_reward == 'divide':
                    steal_multiplier = lambda _: 2 / len(players)

            if do_steal:
                stealables = {p[-1]: tiles.values[p[-1]] * steal_multiplier(p) + potential_loss for p in players if p is not player and any(p)}
            else:
                stealables = []

            print('\nPlayer:     ', player, f'{"+" if player.position > 1 else ""}{player.position}')
            print('Stealable:  ', ', '.join([f'{p.name} [{p[-1]}]' for p in players if not p is player and any(p)]))
            print('Tiles:      ', ' '.join([f'|{t}|' if k else '    ' for t, k in zip(tiles.values, tiles)]))
            print('            ', ' '.join([f'| {s}|' if k else ' __ ' for (_, s), k in zip(tiles.values.items(), tiles)]))
            if not any(tiles):
                break
            dice = Dice()
            roll_n = 0
            while True:
                roll_n += 1
                print(f'{roll_n}. Has:     ', dice, f'({dice.score})')
                roll = dice.roll()
                print('   Roll:    ', roll)
                mask = (dice == 0).astype(int)
                pickups = roll * mask
                if not any(pickups):
                    break_turn()
                    break
                elif np.count_nonzero(pickups) == 1:
                    dice_idx = pickups.argmax()
                    player_dice_idx = dice_idx
                else:
                    if player.name == 'You':
                        player_dice_idx = None
                        while player_dice_idx is None:
                            try:
                                player_dice_idx = int(input("   Pick up:  ")) - 1
                            except ValueError:
                                pass
                            cant = False
                            if player_dice_idx > len(tiles):
                                cant = True
                            elif dice[player_dice_idx] != 0 or roll[player_dice_idx] == 0:
                                cant = True
                            if cant:
                                print('   Cant!')
                                player_dice_idx = None

                    appeals = {}
                    for i, v in enumerate(pickups):
                        if not v:
                            continue
                        new_dice = dice.copy()
                        new_dice[i] = v
                        probs = probabilities[new_dice.key]
                        if do_steal:
                            steal_appeal = pd.Series({f'S{k}': v * stealables[k] for k, v in steal[new_dice.key].items() if k in stealables})
                        appeal = pd.Series([])
                        if any(probs):
                            appeal = pd.Series({k: probs[k] * (v + potential_loss) for k, v in tiles.values.items() if k in probs.index and tiles.is_available(k)})
                        if do_steal and (any(appeal) or any(steal_appeal)):
                            appeal = pd.concat([i for i in (appeal, steal_appeal) if any(i)])
                        elif not any(appeal):
                            continue
                        # Do not optimize probabilities for tiles that we already have
                        new_appeal = pd.Series({k: v for k, v in appeal.items() if int(str(k).replace('S', '')) > dice.score })
                        if any(new_appeal):
                            best_tile = new_appeal.index[np.argmax(new_appeal)]
                        else:
                            best_tile = [k for t, k in zip(tiles, tiles.values) if t and k <= dice.score][-1]
                        if player.name != 'You':
                            print(f'             {v}x{i+1}: ', str(best_tile).rjust(3), f'{(appeal.loc[best_tile]):.4f}')
                        appeals[i] = appeal.loc[best_tile]

                    if not any(appeals):
                        break_turn()
                        break
                    dice_idx = max(appeals, key=appeals.get)
                if player.name == 'You' and np.count_nonzero(pickups) > 1:
                    dice[player_dice_idx] = roll[player_dice_idx]
                    if player_dice_idx != dice_idx:
                        print('   Bot:     ', f'{dice_idx + 1}!!')
                else:
                    dice[dice_idx] = roll[dice_idx]
                # if any(steal_appeal):

                another_turn = True
                if not dice.free_dice:
                    another_turn = False
                else:
                    cur_score = dice.score
                    probs = pd.Series({k: v for k, v in probabilities[dice.key].items() if tiles.is_available(k)})
                    if do_steal:
                        steal_appeal = pd.Series({f'S{k}': v * stealables[k] + potential_loss for k, v in steal[dice.key].items() if k in stealables})
                    appeal = pd.Series([])
                    if any(probs):
                        appeal = pd.Series({k: probs[k] * (v + potential_loss) for k, v in tiles.values.items() if k in probs.index and tiles.is_available(k)})
                    if do_steal and (any(steal_appeal) or any(appeal)):
                        appeal = pd.concat([i for i in (appeal, steal_appeal) if any(i)])
                    if any(appeal):
                        best_tile = appeal.index[appeal.argmax()]
                        target_score = best_tile
                        if isinstance(target_score, str):
                            target_score = int(target_score[1:])
                        if target_score <= dice.score:
                            another_turn = False

                if player.name == 'You':
                    to_pickup = [f'{k} ({v})' for k, v in tiles.values.items() if k <= dice.score and tiles.is_available(k)]
                    if any(to_pickup) and dice.free_dice:
                        print(f'{roll_n}. Has:     ', dice, f'({dice.score}) -> {to_pickup[-1]}')
                        player_another_turn = not any(input('   Quit?     '))
                        if player_another_turn != another_turn:
                            print('   Bot:     ', "Continue!!" if another_turn else "Quit!!")
                        another_turn = player_another_turn

                if not another_turn:
                    print('   Done:    ', dice, f'({dice.score})')
                    if player.name != 'You':
                        for kv in {f'             {k}:      ': f'{v: .3f}' for k, v in appeal.items()}.items(): print(*kv)

                    if target_score in stealables and do_steal:
                        turn_types[game_n].append(2)
                        msg = 'STOLE: '
                        for p in [p for p in players if any(p)]:
                            if p[-1] == target_score:
                                p.pop()
                                break
                    else:
                        tiles.lose(target_score)
                        turn_types[game_n].append(1)
                        msg = 'PICKED:'
                    print(f'   {msg}   |{target_score}|')
                    print(f'             | {tiles.values[target_score]}|')
                    player.append(target_score)
                    break
    try:
        print('FINISHED:')
        for player in players:
            print('            ', player, f'{"+" if player.position > 1 else ""}{player.position}')
        print([f'{p.name} won{"!" if p.name == "You" else "."}' for p in players if p.position > 0][0])

    except IndexError:
        print('The game ended in a draw between', ' and '.join([p.name for p in players if p.position == 0]) + '.')
    if any([p.position > 0 for p in players]):
        win_counter[[p for p in players if p.position > 0][0].name] += 1
n_wins = np.sum(list(win_counter.values()))
print(f'\nTALLY: ({n_wins}/{n_games})')
for player in players:
    print(f"{player.name} won {win_counter[player.name]}/{n_wins} times".ljust(25), f'{win_counter[player.name]/n_wins:.3%}',  ', '.join([f'{k}={v}' for k, v in player.params.items()]))
print('\nOTHER:')
print(n_competitors, f'{np.mean([len(i) for i in turn_types.values()]):.2f}', '+', f'{np.std([len(i) for i in turn_types.values()]):.2f}')















# Define the target length
target_length = 100

# Initialize an empty list to store resampled lists
resampled_lists = []


from scipy.interpolate import interp1d
import matplotlib.ticker as mtick
import seaborn as sns
sns.set_theme(style="whitegrid", palette="pastel")
# Define the target length
target_length = 250
mean_game_len = int(np.mean([len(i) for i in turn_types.values()]).round())
target_indices = np.linspace(1, mean_game_len, target_length)


# Initialize an empty list to store resampled lists
resampled_lists = []

# Resample each list
for lst in turn_types.values():
    # Create the interpolation function
    f = interp1d(np.linspace(1, mean_game_len, len(lst)), lst, kind='nearest')

    # Create interpolation indices for the target length


    # Perform nearest interpolation
    resampled_lst = f(target_indices)

    # Append the resampled list to the result
    resampled_lists.append(resampled_lst)


arr = np.array(resampled_lists)
for i,j in zip([target_indices,
    np.sum(arr == 1, 0)/n_games,
    np.sum(arr == 0, 0)/n_games,
    np.sum(arr == 2, 0)/n_games,], ['Game Duration', 'Pick', 'Lose', 'Steal']): print(f'{j}, ' + ', '.join(map(str, i)))

#
# fig, ax = plt.subplots()
# plt.stackplot(
#     target_indices,
#     np.sum(arr == 1, 0),
#     np.sum(arr == 0, 0),
#     np.sum(arr == 2, 0),
#     labels=['Game Duration', 'Pick', 'Lose', 'Steal']
# )
# ax.set_xlim(0, 100)
# ax.set_ylim(0, 100)
# ax.yaxis.set_major_formatter(mtick.PercentFormatter())
# ax.xaxis.set_major_formatter(mtick.PercentFormatter())
# ax.set_xlabel('Game completion')
# ax.set_ylabel('Share of turns')
# ax.set_title(f'Share of turn outcomes for a {n_games} games with {n_competitors} players')
# fig.legend(loc='lower left')
# fig.show()
#
# fig, ax = plt.subplots()
# ax.imshow(arr)
# ax.set_aspect(10)
# fig.tight_layout()
# fig.show()
# # Now resampled_lists contains lists of length 100