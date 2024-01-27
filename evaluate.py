from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_strategy(
    dataset,
    Model,
    strategy,
    min_date=date(year=2001, month=1, day=1),
    max_date=date(year=2018, month=3, day=4),
    train_period=timedelta(days=365*2),
    test_period=timedelta(days=31*4),
    date_step=timedelta(days=365),
    random_time_periods=False,
    random_trials=100,
    min_bettable_games=50,
    train_data_dropout=0,
    test_data_dropout=0,
    train_swap_prob=0,
    test_swap_prob=0,
    swap_max_timediff=timedelta(days=5),
    swap_max_dist=10,
    verbose=False,
    plot=True,
):
    last_start_date = max_date - train_period - test_period - timedelta(days=1)

    if random_time_periods:
        raise NotImplemented("Have not yet implemented feature for sampling ")

    start_date = min_date

    date_ranges = []
    roi_winnings = []

    while start_date <= last_start_date:
        train_start_date = start_date
        train_end_date = train_start_date + train_period - timedelta(days=1)
        test_start_date = train_end_date + timedelta(days=1)
        test_end_date = train_end_date + test_period

        sample = dataset.take(train_start_date, test_end_date)

        train = sample.take(train_start_date, train_end_date, players_index=sample.players_index)
        train = train.noise(
            dropout=train_data_dropout,
            swap_prob=train_swap_prob,
            swap_max_timediff=swap_max_timediff,
            swap_max_dist=swap_max_dist
        )

        model = Model()
        model.train(train)

        if verbose:
            print('Train loss: ', model.loss())
            print(f'Train accuracy:  {100*model.accuracy():.2f}%')

        test = sample.take(test_start_date, test_end_date, test=True, players_index=sample.players_index)

        games, coeffs = test.get_games_and_coefficients()

        bettable_games = len(test) - np.sum(np.isnan(coeffs).all(axis=-1).any(axis=0))
        if bettable_games < min_bettable_games:
            start_date += date_step
            continue

        test = test.noise(
            dropout=test_data_dropout,
            swap_prob=test_swap_prob,
            swap_max_timediff=swap_max_timediff,
            swap_max_dist=swap_max_dist
        )

        bets = strategy(model, test)
        trial_winnings = test.winnings(bets, model)

        roi_winnings.append(list(trial_winnings))
        date_ranges.append([test_start_date, test_end_date])

        if verbose:
            print(test_start_date.isoformat(), test_end_date.isoformat())
            print('Final ROI:')
            print(f'{100*trial_winnings[-1]:.2f}%')
            print('Model test accuracy:')
            print(f'{100*model.accuracy(test):.2f}%')
            print('################################################################################')

        start_date += date_step

    if plot:
        print(len(roi_winnings))
        x = np.concatenate([ np.arange(len(curr_winnings)) for curr_winnings in roi_winnings ])
        y = [ x*100 for curr_winnings in roi_winnings for x in curr_winnings ]
        c = [ i for i, curr_winnings in enumerate(roi_winnings) for _ in curr_winnings ]
        plt.xlabel('Number of games played in period')
        plt.ylabel('ROI (%) - Return Over Investment')
        plt.ylim((-100, 200))
        scatter = plt.scatter(x, y, c=c, s=5)
        handles = scatter.legend_elements()[0]
        plt.legend(
            handles=handles,
            labels=[f'{d1.isoformat()} - {d2.isoformat()}' for d1, d2 in date_ranges],
            loc="upper right",
            title="Betting period"
        )
        plt.show()

    return roi_winnings
