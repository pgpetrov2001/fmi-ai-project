from datetime import date, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import calculate_returns, rolling_returns

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
    plot_x='time',
    plot_y='roi',
    save_plot=None,
):
    valid_plot_x_options = ['games_played', 'bets_placed']
    if plot_x not in valid_plot_x_options:
        raise ValueError(f'Invalid value "{plot_x}" for argument plot_x. Valid options are "{valid_plot_x_options}"')

    valid_plot_y_options = ['rolling_roi', 'roi']
    if plot_y not in valid_plot_y_options:
        raise ValueError(f'Invalid value "{plot_y}" for argument plot_y. Valid options are "{valid_plot_y_options}"')

    plot_x_option_label = {
        'games_played': 'Number of games played in period',
        'bets_placed': 'Number of bets placed'
    }
    plot_y_option_label = {
        'roi': 'ROI(%) - Return Over Investment',
        'rolling_roi': 'Rolling ROI(%) - Rolling Return Over Investment',
    }
    plot_x_option_xlim = {
        'bets_placed': (0, 150),
        'games_played': (0, 1200),
    }

    last_start_date = max_date - train_period - test_period - timedelta(days=1)

    if random_time_periods:
        raise NotImplemented("Have not yet implemented feature for sampling ")

    start_date = min_date

    returns = []
    bets = []
    date_ranges = []

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

        trial_bets = strategy(model, test)
        trial_returns, trial_bets = calculate_returns(test, trial_bets)

        returns.append(trial_returns)
        bets.append(trial_bets)
        date_ranges.append([test_start_date, test_end_date])

        if verbose:
            print(test_start_date.isoformat(), test_end_date.isoformat())
            print('Final ROI:')
            print(f'{100*trial_returns[-1]/trial_bets.sum():.2f}%')
            print('Model test accuracy:')
            print(f'{100*model.accuracy(test):.2f}%')
            print('################################################################################')

        start_date += date_step

    if plot:
        if plot_y == 'roi':
            y = [curr_returns.cumsum() / curr_bets.sum() for curr_returns, curr_bets in zip(returns, bets)]
        elif plot_y == 'rolling_roi':
            y = [rolling_returns(curr_returns, curr_bets) for curr_returns, curr_bets in zip(returns, bets)]

        if plot_x == 'bets_placed':
            bet_masks = [ curr_bets != 0 for curr_bets in bets ]
            bets = [ curr_bets[bet_mask] for curr_bets, bet_mask in zip(bets, bet_masks) ]
            y = [ curr_y[bet_mask] for curr_y, bet_mask in zip(y, bet_masks) ]

        x = np.concatenate([ np.arange(len(curr_bets)) for curr_bets in bets ])
        y = [ val*100 for curr_y in y for val in curr_y ]
        c = [ i for i, curr_bets in enumerate(bets) for _ in curr_bets ]
        plt.title(f'Model: {Model.description()}. Strategy: {strategy.description()}')
        plt.xlabel(plot_x_option_label[plot_x])
        plt.ylabel(plot_y_option_label[plot_y])
        plt.xlim(plot_x_option_xlim[plot_x])
        plt.ylim((-100, 200))
        scatter = plt.scatter(x, y, c=c, s=5)
        handles = scatter.legend_elements()[0]
        plt.legend(
            handles=handles,
            labels=[f'{d1.isoformat()} - {d2.isoformat()}' for d1, d2 in date_ranges],
            loc="upper right",
            title="Betting period"
        )
        if save_plot is not None:
            plt.savefig(save_plot)
        plt.show()
