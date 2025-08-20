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

mult_perfs = []
mults = np.arange(1.0, 3.5, 0.5)
ns_players = range(2, 8)

for mult in mults:
    mult_perf = []
    for n_players in ns_players:
        target = pd.read_csv(Path.cwd() / f'df_games-10000_players-{n_players}_mult-2_steal-yes.csv')
        target_win_rate = (target.winner == 0).sum() / target.winner.notna().sum()
        res = pd.read_csv(Path.cwd() / f'df_games-100_players-{n_players}_mult-{mult}_steal-yes.csv')
        perf = (res.winner == 0).sum() / res.winner.notna().sum() - target_win_rate
        mult_perf.append(perf)
    mult_perfs.append(mult_perf)
mult_perfs = np.array(mult_perfs)

fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.grid(visible=False)
h = ax.imshow(mult_perfs, cmap='Blues',)
ax.set_xticks(np.arange(6), ns_players)
ax.set_yticks(np.arange(5), mults)

ax.set_xlabel('Number of Players')
ax.set_ylabel('Multiplier')
cbar = fig.colorbar(h)
cbar.set_label('$\Delta$ win-rate')
fig.show()
