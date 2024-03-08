from utils import Tiles, Dice, Player, pickle_in
import numpy as np
import pandas as pd

minimal_probs = pickle_in('minimal.pkl')
exact_probs = pickle_in('exactly.pkl')
n_competitors = 3
n_games = 1
players = [Player() for _ in range(n_competitors)]


def log(key, i=1, *values):
    print(f'{" " * 4 * i}{key}:'.ljust(16), *values)


def break_turn():
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


def prob2appeal(_dice: Dice, steal=False):
    all_appeals = pd.Series(data=[.0] * len(available_tiles), index=available_tiles)
    val = tiles.values
    if steal:
        source = exact_probs
        availability = stealables
        token = 'S'
        mult = 2
    else:
        source = minimal_probs
        availability = tiles
        mult = 1
        token = 'P'
    available_appeals = pd.Series({f'{token}{tile}': prob * val[tile] * mult for tile, prob in source[_dice.key].items() if tile in availability})
    all_appeals.update(available_appeals)
    return all_appeals


alice_counter = 0
for game_n in range(n_games):
    tiles = Tiles()
    [player.reset() for player in players]
    while any(tiles):
        for player in players:
            # Set up the game
            potential_loss = 0 if not any(player) else player[-1]
            stealables = {p[-1]: tiles.values[p[-1]] * 2 + potential_loss for p in players if p is not player and any(p)}
            available_tiles = list([f'P{s}' for s in tiles.values]) + list([f'S{s}' for s in stealables])
            print('\nPlayer:     ', player, f'{"+" if player.position > 1 else ""}{player.position}')
            print('Stealable:  ', ', '.join([f'{p.name} [{p[-1]}]' for p in players if not p is player and any(p)]))
            if not any(tiles):
                break
            dice = Dice()
            roll_n = 0

            while True:
                #  Print start of a roll
                roll_n += 1
                print(f'{roll_n}. Has:     ', dice, f'({dice.score})')

                # Roll the dice
                roll = dice.roll()
                print('   Roll:    ', roll)

                # Check what die faces can be picked up
                mask = (dice == 0).astype(int)
                pickups = roll * mask

                # If no faces can be picked up the turn ends
                if not any(pickups):
                    break_turn()
                    break

                # Only one pickup is possible, take it
                if np.count_nonzero(pickups) == 1:
                    dice_idx = pickups.argmax()
                else:
                    appeals = {}
                    for idx, freq in [f for f in enumerate(pickups) if f[1]]:
                        # Imagine the new collection
                        new_dice = dice.copy()
                        new_dice[idx] = freq
                        a = prob2appeal(new_dice)
                        b = prob2appeal(new_dice, steal=True)
                        # Imagine what we could pick up
                        pick_appeal = minimal_probs[new_dice.key]
                        new_probabilities.update()
                        # Imagine its minimal_probs
                        probs = minimal_probs[new_dice.key]
                        steal_appeal = pd.Series({f'S{tile}': prob * stealables[tile] for tile, prob in exact_probs[new_dice.key].items() if tile in stealables})
                        appeal = pd.Series([])
                        if any(probs):
                            appeal = pd.Series({k: probs[k] * (v + potential_loss) for k, v in tiles.values.items() if k in probs.index and tiles.is_available(k)})
                        if any(appeal) or any(steal_appeal):
                            appeal = pd.concat([i for i in (appeal, steal_appeal) if any(i)])
                        elif not any(appeal):
                            continue
                        # Do not optimize minimal_probs for tiles that we already have
                        new_appeal = pd.Series({k: v for k, v in appeal.items() if int(str(k).replace('S', '')) > dice.score })
                        if any(new_appeal):
                            best_tile = new_appeal.index[np.argmax(new_appeal)]
                        else:
                            best_tile = [k for t, k in zip(tiles, tiles.values) if t and k <= dice.score][-1]
                        print(f'             {freq}x{idx + 1}: ', str(best_tile).rjust(3), f'{(appeal.loc[best_tile]):.4f}')
                        appeals[idx] = appeal.loc[best_tile]

                    if not any(appeals):
                        break_turn()
                        break
                    dice_idx = max(appeals, key=appeals.get)
                dice[dice_idx] = roll[dice_idx]

                another_turn = True
                if not dice.free_dice:
                    another_turn = False
                else:
                    cur_score = dice.score
                    probs = pd.Series({k: v for k, v in minimal_probs[dice.key].items() if tiles.is_available(k)})
                    steal_appeal = pd.Series({f'S{k}': v * stealables[k] + potential_loss for k, v in exact_probs[dice.key].items() if k in stealables})
                    appeal = pd.Series([])
                    if any(probs):
                        appeal = pd.Series({k: probs[k] * (v + potential_loss) for k, v in tiles.values.items() if k in probs.index and tiles.is_available(k)})
                    if any(steal_appeal) or any(appeal):
                        appeal = pd.concat([i for i in (appeal, steal_appeal) if any(i)])
                    if any(appeal):
                        best_tile = appeal.index[appeal.argmax()]
                        target_score = best_tile
                        if isinstance(target_score, str):
                            target_score = int(target_score[1:])
                        if target_score <= dice.score:
                            another_turn = False
                if not another_turn:
                    print('   Done:    ', dice, f'({dice.score})')
                    for kv in {f'             {k}:      ': f'{v: .3f}' for k, v in appeal.items()}.items(): print(*kv)

                    if target_score in stealables:
                        msg = 'STOLE: '
                        for p in [p for p in players if any(p)]:
                            if p[-1] == target_score:
                                p.pop()
                                break
                    else:
                        tiles.lose(target_score)
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
    if [p.position > 0 for p in players if p.name == 'Alice'][0]:
        alice_counter += 1

samples = 50
foe = [[52.99, 47.01, ],
       [37.59, 33.88, 28.53, ],
       [28.61, 26.27, 24.58, 20.54, ],
       [23.58, 23.02, 20.24, 17.37, 15.80, ],
       [20.47, 19.63, 17.50, 16.45, 14.11, 11.84, ],
       [18.59, 16.99, 15.78, 13.41, 13.31, 11.86, 10.05, ], ]

baa = [[51.61, 48.40, ],
       [34.51, 34.33, 31.17, ],
       [26.74, 24.88, 24.49, 23.90, ],
       [20.30, 20.64, 19.99, 20.16, 18.92, ],
       [17.27, 16.78, 17.72, 16.31, 15.67, 16.25, ],
       [14.74, 14.54, 14.94, 13.81, 14.18, 13.79, 14.00, ]]


import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
sns.set_theme(style="whitegrid", palette="pastel")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Fira Sans"]
c=1
for foo in [foe, baa]:
    bar = [np.array(f) - 100/i for i, f in enumerate(foo, 2)]
    bars = []
    x = np.linspace(0, 1, samples)
    for fp in bar:
        xp = np.linspace(0, 1, len(fp))
        step = xp[1] - xp[0]
        xp = np.array([x[0] - step/2, *list(xp), xp[-1] + step/2])
        xp -= xp.min()
        xp /= xp.max()
        fp = np.array([fp[0], *list(fp), fp[-1]])
        f = interp1d(xp, fp, kind='nearest')
        bars.append(f(x))
    arrs = np.array(bars)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    h = ax.imshow(arrs/100, cmap='Blues', clim=(-0.045, 0.045))
    ax.set_aspect(samples/7)
    ax.grid(visible=False)

    # Set x ticks
    ax.set_xticks([0, samples // 2, samples])
    ax.set_xticklabels(['First', 'Middle', 'Last'])

    # Set y ticks
    ax.set_yticks(range(6))
    ax.set_yticklabels(range(2, 8))
    ax.set_title(f'First-play win-rate w/{"o" if c != 2 else ""} stealing')
    # Set labels
    ax.set_ylabel('Number of players')
    ax.set_xlabel('Starting position')
    # Set colorbar label
    cbar = fig.colorbar(h)
    cbar.set_label('$\Delta$ win-rate')

    # Set percentages to color bar ticks
    cbar.ax.set_yticklabels(['{:.0f}%'.format(i * 100) for i in cbar.get_ticks()])

    # Show plot
    fig.savefig(f'Test{c}.svg')
    fig.tight_layout()
    fig.show()
    c+=1
