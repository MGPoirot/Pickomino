from scipy.stats import norm
from play import roll_dice, analyze_turn, count_score, n_dice, get_score_probabilities
import numpy as np


def run_simulation(test, evals=10_000):
    minimum, start_collection = test
    expectation = get_score_probabilities(np.array(start_collection)).loc[minimum]

    wins = []
    for i in range(evals):
        collection_state = np.array(start_collection)
        while True:
            roll_state = roll_dice(collection_state)
            info = analyze_turn(collection_state, roll_state)
            if not any(info):
                wins.append(False)
                break
            picked_idx = info.loc[minimum].idxmax(axis=1).values[0] - 1
            collection_state[picked_idx] = roll_state[picked_idx]
            score = count_score(collection_state)
            if score >= minimum:
                wins.append(True)  # We achieved the desired score
                break
            elif collection_state.sum() == n_dice:
                wins.append(False)  # No dice left
                break
    n = len(wins)
    p = np.sum(wins) / n
    se = np.sqrt(p * (1 - p) / n)  # Standard error
    z = norm.ppf((1 + 0.95) / 2)  # Z-score
    err = z * se
    print(f"{minimum} from start {start_collection} "
          f"predicted win rate is{expectation: .3%}, "
          f"realised win rate is{p: .3%} (CI ={p-err: .3%} -{p+err: .3%})",
          f"{'SUCCESS' if expectation > p-err and expectation < p+err else 'FAIL'}")



if __name__ == '__main__':
    # However, when we simulate we see that the score 21 is achieved 86.330% (CI=85.657%-87.003%) of the time
    tests = [
        (21, (0, 0, 0, 7, 0, 0), 1/6),
        (21, (0, 0, 0, 6, 0, 0), ((5+6)/36) + (20/36)/6),
        (21, (0, 6, 0, 0, 0, 1), 2/6),
        (29, (0, 1, 4, 0, 0, 1), 1/36),
        (21, (0, 0, 0, 0, 0, 0)),
        (22, (0, 0, 0, 0, 0, 0)),
        (23, (0, 0, 0, 0, 0, 0)),
        (24, (0, 0, 0, 0, 0, 0)),
        (25, (0, 0, 0, 0, 0, 0)),
        (26, (0, 0, 0, 0, 0, 0)),
        (27, (0, 0, 0, 0, 0, 0)),
        (28, (0, 0, 0, 0, 0, 0)),
        (29, (0, 0, 0, 0, 0, 0)),
        (30, (0, 0, 0, 0, 0, 0)),
        (31, (0, 0, 0, 0, 0, 0)),
        (32, (0, 0, 0, 0, 0, 0)),
        (33, (0, 0, 0, 0, 0, 0)),
        (34, (0, 0, 0, 0, 0, 0)),
        (35, (0, 0, 0, 0, 0, 0)),
        (36, (0, 0, 0, 0, 0, 0)),
        # (21, (1, 0, 0, 0, 0, 0)),
        # (21, (0, 1, 0, 0, 0, 0)),
        # (21, (0, 0, 1, 0, 0, 0)),
        # (21, (0, 0, 0, 1, 0, 0)),
        # (21, (0, 0, 0, 0, 1, 0)),
        # (21, (0, 0, 0, 0, 0, 1)),
        # (21, (0, 0, 0, 1, 0, 0)),
        # (21, (0, 0, 0, 2, 0, 0)),
        # (21, (0, 0, 0, 3, 0, 0)),
        # (21, (0, 0, 0, 4, 0, 0)),
        # (21, (0, 0, 0, 5, 0, 0)),
        # (21, (0, 0, 0, 6, 0, 0)),
        # (21, (0, 0, 0, 7, 0, 0)),
        # (22, (1, 0, 0, 0, 0, 4), 0.907),
        # (23, (2, 0, 0, 0, 0, 4), 1-4/36),
        # (21, (0, 0, 0, 0, 0, 5), 1-1/6**3),
    ]
    sorted_tests = sorted(tests, key=lambda x: x[0])
    for test in sorted_tests:
        run_simulation(test, evals=1000)
