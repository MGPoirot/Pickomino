from utils import Tiles, Dice, Players, pickle_in, Player
import numpy as np
import pandas as pd


def _strip_s(tile_number: str | int) -> int:
    # Turns "S25" into 25
    return int(str(tile_number).replace('S', ''))


def break_turn(game_n: int, tiles: Tiles, player: Player):
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


def play_single_game(game_n: int = 1) -> None:
    """
    This function does not export anything, instead it assigns directory
     to turn_type and win_counter directories in the global scope.
    :param game_n:
    :return:
    """
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
                    break_turn(game_n, tiles, player)
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
                        # Do not optimize probabilities for tiles_p that we already have
                        new_appeal = pd.Series({k: v for k, v in appeal.items() if int(str(k).replace('S', '')) > dice.score })
                        if any(new_appeal):
                            best_tile = new_appeal.index[np.argmax(new_appeal)]
                        else:
                            best_tile = [k for t, k in zip(tiles, tiles.values) if t and k <= dice.score][-1]
                        if player.name != 'You':
                            print(f'             {v}x{i+1}: ', str(best_tile).rjust(3), f'{(appeal.loc[best_tile]):.4f}')
                        appeals[i] = appeal.loc[best_tile]

                    if not any(appeals):
                        break_turn(game_n, tiles, player)
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
                        for kv in {f'             {(str(k) + ":").rjust(4)}     ': f'{v: .3f}' for k, v in appeal.items()}.items(): print(*kv)

                    # Cap appeal by the value of dice thrown
                    capped_appeal = appeal[[tile_value <= dice.score for tile_value in map(_strip_s, appeal.index)]]

                    # If no tile can be picked up with the value of dice thrown, break the turn
                    if not any(capped_appeal):
                        break_turn(game_n, tiles, player)
                        return

                    # Get the highest tile with the highest achieved appeal
                    target_score = _strip_s(capped_appeal.index[capped_appeal.eq(capped_appeal.max())].max())

                    if target_score in stealables and do_steal:
                        msg = 'STOLE: '
                        # Add a steal to the turn types
                        turn_types[game_n].append(2)
                        for p in [p for p in players if any(p)]:
                            if p[-1] == target_score:
                                p.pop()
                                break
                    else:
                        msg = 'PICKED:'
                        # Add a table pick to the turn types
                        turn_types[game_n].append(1)
                        tiles.lose(target_score)
                    print(  f'   {msg}   |{target_score}|')
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


### PARAMETERS
human_player = True
n_players = 4
n_games = 10
do_steal = True


# Initialization
if human_player:
    print(
        f"""
        Let's get started with playing Pickomino!
        I assume you are familiar with the rules of this dice game. This game can be played entirely with the num pad.
        You will be playing against {n_players - 1} computer players that can{'' if do_steal else ' not'} steal tiles.
        Let's first look at how to read the state of the game, then at picking up dice, and then at deciding to pick up
        a tile.
        
        GAME STATE:
            Player:      You ()  -2
            Stealable:   Bob [26], Charlie [25]
            Tiles:       |21| |22| |23| |24|           |27| |28| |29| |30| |31| |32| |33| |34| |35| |36|
                         | 1| | 1| | 1| | 1|  __   __  | 2| | 2| | 3| | 3| | 3| | 3| | 4| | 4| | 4| | 4|
        This game has just started. You currently own no tiles, and your competitor Alice does not have a tile, but Bob 
        and Charlie do. This way, you are 2 points behind the game's leader. You can steal their tiles and of course 
        pick up tiles from the table.
        
        Next to the player's name you see parameters you provided to the player upon creation. This way, you can easily
        see of modifiers you've implemented improve player behaviour. For example:
        >>> print(Players([('multiplier': 4), ()]))
        ... [Alice ('multiplier': 4) , Bob () ]
        PICK UP DICE:
            1. Has:      [0 0 0 0 0 0] (0)
               Roll:     [2 1 1 1 2 1]
               Pick up:  >? 5
               Bot:      3!!
            2. Has:      [0 0 0 0 2 0] (0)
               Roll:     [3 0 1 2 0 0]
               Pick up:  >? 4
        At the beginning of your first turn, you have no dice collected. 
        In your first roll, you roll two ones, two fives and one of everything else. 
        You choose to pick up the fives and will see that you start the second roll with two fives.
        If the bots disagree with your choice they will let you know ('Bot:      3!!' means that they would have 
        expected you to pick a 3, which is made up in this case).
        The value of your dice is still zero (0) because you do no have a Worm dice.
        
        PICK UP A TILE:
        3. Has:      [0 0 0 2 2 0] (0)
           Roll:     [1 0 1 1 0 1]
           Pick up:  >? 6
        3. Has:      [0 0 0 2 2 1] (23) -> 23 (1)
           Quit?     >? 1
           Bot:      Continue!!
           Done:     [0 0 0 2 2 1] (23)
           STOLE:    |26|
                     | 2|
           Player:      Alice ()  -2
           Stealable:   You [26], Charlie [25]
           Tiles:       |21| |22| |23| |24|           |27| |28| |29| |30| |31| |32| |33| |34| |35| |36|
                        | 1| | 1| | 1| | 1|  __   __  | 2| | 2| | 3| | 3| | 3| | 3| | 4| | 4| | 4| | 4|
        In your third roll, you pick up a Worm. Now the value of your collected dice is 23 and you are asked if you 
        want to pick up a tile (Quit) or continue. 
        You enter any key to Quit.
        The Bots again disagree with you and would have expected you to "Continue!!".
        
        Please only enter numbers, this code is not bombproof and will break if you enter nothing or strings.
        Have fun!
        """
    )
players = Players([{'name': 'You'}] + [{}] * (n_players - 1)) if human_player else Players(n=n_players)
probabilities = pickle_in('precalculated_chances/minimal.pkl')
steal = pickle_in('precalculated_chances/exactly.pkl')

# Preallocate dicts to store game results to
turn_types = {}
win_counter = {p.name: 0 for p in players}

# Play N games
for game_number in range(n_games):
    play_single_game(game_number)

# Print results
n_wins = np.sum(list(win_counter.values()))
print(f'\nTALLY: ({n_wins}/{n_games})')
for player in players:
    print(f"{player.name} won {win_counter[player.name]}/{n_wins} times".ljust(25), f'{win_counter[player.name]/n_wins:.3%}',  ', '.join([f'{k}={v}' for k, v in player.params.items()]))
print('\nN_PLAYERS N_TURNS + SD:')
print(str(n_players).rjust(9), f'{np.mean([len(i) for i in turn_types.values()]):.2f}'.rjust(7), '+', f'{np.std([len(i) for i in turn_types.values()]):.2f}')