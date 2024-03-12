from utils import Tiles, Dice, Player, Players, pickle_in, log
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
from pathlib import Path
np.random.seed(0)


def break_turn(player, tiles):
    # Check if the player has anything to lose
    if any(player):
        # Check if the table has any tiles to flip
        if any(tiles):
            # What is the rule when returning a tile, and nothing is there?
            last_tile = [v for k, v in zip(tiles, tiles.values) if k][-1]
            tiles.lose(last_tile)
        lost_tile = player.pop()
        tiles.gain(lost_tile)
        tile_worth = tiles.values[lost_tile]
        return lost_tile, tile_worth
    else:
        return '', ''


def appeal_wrapper(tiles, players, player, multiplier=2):
    potential_loss = 0 if not any(player) else player[-1]
    if do_steal:
        tiles_s = {p[-1]: tiles.values[p[-1]] * multiplier + potential_loss for p in players if p is not player and any(p)}
    else:
        tiles_s = {}
    available_tiles = list([f'P{s}' for s in tiles.values]) + list([f'S{s}' for s in tiles_s])

    def appeal_calc(dice: Dice):
        appeals = None
        for (source, availability, mult, token) in (minimal_probs, tiles, 1.0, 'P'), (exact_probs, tiles_s, 2.0, 'S'):
            all_appeals = pd.Series(data=[.0] * len(available_tiles), index=available_tiles)
            available_appeals = pd.Series({f'{token}{tile}': prob * tiles.values[tile] * mult for tile, prob in source[dice.key].items() if tile in availability})
            all_appeals.update(available_appeals)
            if appeals is None:
                appeals = all_appeals
            else:
                appeals += all_appeals
        return appeals
    return appeal_calc


def play_game(players: List[Player], logger=log, multiplier=2.0):
    # Set up the game
    tiles = Tiles()
    [player.reset() for player in players]
    turn_outcomes = []
    while any(tiles):
        for player in players:
            # Start a new turn for the player
            exit_reason = None
            dice = Dice()
            roll_n = 0
            calc_appeal = appeal_wrapper(tiles, players, player, multiplier=player('multiplier', multiplier))
            # Report
            no_logger = logger('\nPlayer', '', player, f'{"+" if player.position > 1 else ""}{player.position}', i=0)
            if do_steal:
                logger('Stealable', ', '.join([f'{p.name} [{p[-1]}]' for p in players if not p is player and any(p)]), i=0)
            if not no_logger:
                print(tiles)

            while dice.free_dice:
                #  Print start of a roll
                roll_n += 1
                logger(f'{roll_n}.  Has', dice, f'({dice.score})', i=0)
                # Roll the dice
                roll = dice.roll()
                logger('Roll', roll)
                # Check what die faces can be picked up
                mask = (dice == 0).astype(int)
                pickups = roll * mask

                if not any(pickups):
                    # Player has gone broke
                    exit_reason = 'broken'
                    break

                # Only one pickup is possible, take it
                if np.count_nonzero(pickups) == 1:
                    dice_idx = pickups.argmax()
                    logger(f'{pickups[dice_idx]}x{dice_idx + 1}', 'Forced', i=2)
                else:
                    appeals = {}
                    for idx, freq in [f for f in enumerate(pickups) if f[1]]:
                        # Imagine the new collection
                        new_dice = dice.copy()
                        new_dice[idx] = freq
                        try:
                            appeal = calc_appeal(new_dice)
                        except UnboundLocalError as err:
                            print(err)
                            continue
                        # Do not optimize minimal_probs for tiles_p that we already have
                        best_tile = appeal.index[appeal.argmax()]
                        logger(f'{freq}x{idx + 1}', str(best_tile).rjust(3), f'{(appeal.max()): .4f}', i=2)
                        appeals[idx] = appeal.loc[best_tile]
                    dice_idx = max(appeals, key=appeals.get)

                # Add to collection
                dice[dice_idx] = roll[dice_idx]

                # Evaluate collection
                appeal = calc_appeal(new_dice)

                if not any(appeal):
                    # No available tile can be reached anymore
                    exit_reason = 'hopeless'
                    break

                # Decide what the best tile is after pickup
                best_tile = appeal.index[appeal.argmax()]

                # Finish rolling
                if dice.score >= int(best_tile[1:]):
                    exit_reason = 'choice'
                    break

            logger(f'Exit {exit_reason}', dice, f'({dice.score})')
            if exit_reason != 'choice':
                available_pickups = []
            else:
                # Player exited the turn handle tiles_p
                available_pickups = pd.Series({k: v for k, v in calc_appeal(dice).items() if
                                               (k[0] == 'P' and int(k[1:]) <= dice.score) or
                                               (k[0] == 'S' and int(k[1:]) == dice.score)})

            if not any(available_pickups):
                msg = 'LOST'
                turn_outcomes.append(0)
                tile_value, tile_worth = break_turn(player, tiles)
            else:
                for kv in {k: f'{v: .3f}' for k, v in appeal.items()}.items():
                    logger(*kv, i=2)
                picked_tile = available_pickups[available_pickups == available_pickups.max()].index[-1]
                tile_source, tile_value = picked_tile[0], int(picked_tile[1:])
                tile_worth = tiles.values[tile_value]
                if tile_source == 'P':
                    msg = 'PICKED'
                    turn_outcomes.append(1)
                    prev_owner = tiles
                else:
                    msg = 'STOLE '
                    turn_outcomes.append(2)
                    prev_owner = [p for p in players if any(p) and p[-1] == tile_value][0]
                player.append(prev_owner.transfer(tile_value))
            logger(msg, f'|{str(tile_value).rjust(2)}|')
            logger('', f'|{str(tile_worth).rjust(2)}|')

            # If not tiles are left the game is over
            if not tiles.free_tiles:
                break

    winner = [i for i, p in enumerate(players) if p.position > 0]
    if len(winner) > 0:
        winner = winner[0]
    else:
        winner = None
    return winner, tuple(turn_outcomes)


# Parameters
n_games = 10_000
do_steal = True
verbose = False

# Initialization
minimal_probs = pickle_in('minimal.pkl')
exact_probs = pickle_in('exactly.pkl')
log_method = log if verbose else lambda *args, i=None: True


for mult in [2.0, ]:  # np.arange(1.0, 3.5, 0.5):
    for n_players in [2, ]:  # range(2, 8):
        friends = Players([{'multiplier': mult}] + [{}] * (n_players - 1))
        df = pd.DataFrame(columns=['winner', 'turn_outcome'])
        df.index.name = 'game_n'
        fname = Path(
            f'df_'
            f'games-{n_games}_'
            f'players-{n_players}_'
            f'mult-{mult}_'
            f'steal-{"yes" if do_steal else "no"}.csv'
        )
        for game_n in tqdm(range(n_games)):
            try:
                data = play_game(
                    players=friends,
                    logger=log_method,
                )
            except Exception as e:
                print(e)
                continue
            df.loc[game_n] = data
        df.to_csv(fname)

