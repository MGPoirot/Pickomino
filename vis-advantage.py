import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
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

# PARAMS:
samples = 50

# FAKE FILE: df_games-10000_players-6_mult-2_steal-no.csv
for do_steal in ('no', 'yes'):
    fname = 'honest' if do_steal == 'no' else 'steal'

    plot_data = []
    for n_players in range(2, 8):
        res = pd.read_csv(Path.cwd() / f'df_games-10000_players-{n_players}_mult-2_steal-{do_steal}.csv')
        players = res[res.winner.notna()].winner.unique()
        players.sort()
        plot_data.append([np.array((res.winner == player).sum() / res.winner.notna().sum()) - 1 / n_players for player in players])

    bars = []
    x = np.linspace(0, 1, samples)
    for fp in plot_data:
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
    h = ax.imshow(arrs, cmap='Blues', clim=(-0.045, 0.045))
    ax.set_aspect(samples/7)
    ax.grid(visible=False)

    # Set x ticks
    ax.set_xticks([samples / 7 / 2, samples - (samples / 7 / 2)])
    ax.set_xticklabels(['First', 'Last'])

    # Set y ticks
    ax.set_yticks(range(6))
    ax.set_yticklabels(range(2, 8))
    ax.set_title(f'$\Delta$ win-rate w/{"o" if fname == "honest" else ""} stealing')
    # Set labels
    ax.set_ylabel('Number of players')
    ax.set_xlabel('Starting position')
    # Set colorbar label
    cbar = fig.colorbar(h)

    # Set percentages to color bar ticks
    cbar.ax.set_yticklabels(['{:.0f}%'.format(i * 100) for i in cbar.get_ticks()])

    # Show plot
    fig.savefig(f'first-{fname}.svg', transparent=True)
    fig.tight_layout()
    fig.show()
