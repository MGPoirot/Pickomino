"""
Visualize the completion of the game after N turns.
"""

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


fig, ax = plt.subplots()
plt.stackplot(
    target_indices,
    np.sum(arr == 1, 0),
    np.sum(arr == 0, 0),
    np.sum(arr == 2, 0),
    labels=['Game Duration', 'Pick', 'Lose', 'Steal']
)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlabel('Game completion')
ax.set_ylabel('Share of turns')
ax.set_title(f'Share of turn outcomes for a {n_games} games with {n_players} players')
fig.legend(loc='lower left')
fig.show()

fig, ax = plt.subplots()
ax.imshow(arr)
ax.set_aspect(10)
fig.tight_layout()
fig.show()
# Now resampled_lists contains lists of length 100