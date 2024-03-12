import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path
import pandas as pd

sns.set_theme(style="whitegrid", palette="pastel")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Fira Sans"]

COLOR = 'gray'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR

target_length = 500


for do_steal in ('yes', ):
    for n_players in range(2, 8):
        figname = 'turnoutcome_players-{}.svg'
        filepath = Path.cwd() / f'df_games-10000_players-{n_players}_mult-2_steal-{do_steal}.csv'
        if not filepath.is_file():
            continue
        turn_types = pd.read_csv(filepath, converters={"turn_outcome": lambda x: tuple(map(int, x[1:-1].split(', ')))})

        n_games = len(turn_types)

        # Initialize an empty list to store resampled lists
        resampled_lists = []
        mean_game_len = np.mean(list(map(len, turn_types.turn_outcome))).round().astype(int)
        target_indices = np.linspace(1, mean_game_len, target_length)

        # Initialize an empty list to store resampled lists
        resampled_lists = []

        # Resample each list
        for lst in turn_types.turn_outcome.values:
            # Create the interpolation function
            f = interp1d(np.linspace(1, mean_game_len, len(lst)), lst, kind='nearest')

            # Create interpolation indices for the target length


            # Perform nearest interpolation
            resampled_lst = f(target_indices)

            # Append the resampled list to the result
            resampled_lists.append(resampled_lst)


        arr = np.array(resampled_lists)
        for i,j in zip([target_indices,
            np.sum(arr == 1, 0) / n_games,
            np.sum(arr == 0, 0) / n_games,
            np.sum(arr == 2, 0) / n_games,], ['Game Duration', 'Pick', 'Lose', 'Steal']):
            pass # print(f'{j}, ' + ', '.join(map(str, i)))

        fig, ax = plt.subplots(figsize=(6, 4))
        plt.stackplot(
            target_indices,
            np.sum(arr == 1, 0) / n_games * 100,
            np.sum(arr == 0, 0) / n_games * 100,
            np.sum(arr == 2, 0) / n_games * 100,
            labels=['Pick', 'Lose', 'Steal'],
            colors=plt.cm.Blues(np.linspace(0.2, 0.8, 3)),
        )
        ax.set_xlim(1, 51)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xlabel(' ')
        # ax.set_ylabel('Share of turn outcomes')
        ax.set_title(f'Turn outcomes for a game with {n_players} players')
        ax.grid(visible=False)
        fig.legend(loc='lower center', bbox_to_anchor=(0.56, -0.04), ncol=3, frameon=False)
        fig.savefig(figname.format(n_players), transparent=True)
        fig.show()
