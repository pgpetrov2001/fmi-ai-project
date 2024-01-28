import numpy as np
import sys
import pandas as pd

class ExpectedValueCutoff:
    """
    There is an allocation of 1$ for each bet.
    If the expected value of the ROI of this bet is above a certain fixed cutoff percentage,
    according to our estimate of the probability of our preferred player winning,
    then we place the bet of 1$ on our preferred player, otherwise we don't.
    This strategy depends on a cutoff paramater specifying cutoff value for
    estimate of expected returns for the particular bet, expressed as a ratio - default is 0.2
    """

    def __init__(self, threshold=0.2):
        self.threshold = 1 + threshold

    def __call__(self, model, dataset):
        games, coeffs = dataset.get_games_and_coefficients()
        predictions = model.predict(games) # ngames
        favorites = (predictions < 0.5).astype(np.int32) # ngames
        c = coeffs[:, np.arange(games.shape[0]), favorites] # nbookmakers x ngames
        predictions[predictions < 0.5] = 1-predictions[predictions < 0.5]
        expected_values = predictions * c - (1-predictions) # nbookmakers x ngames
        mask = (predictions != 0.5) * (~np.isnan(expected_values)) * (expected_values > self.threshold) # nbookmakers x ngames
        bets = np.zeros(coeffs.shape, dtype=np.float32) # nbookmakers x ngames x 2

        if mask.sum() == 0:
            # decided to not make any bets (most likely due to missing data)
            return bets

        favorites = np.repeat([favorites], coeffs.shape[0], axis=0)
        bets[mask, favorites[mask]] = 1
        return bets

    def description(self):
        return f'EV cutoff={100*(self.threshold-1):.2f}%'

class EveryGameBetModelFavorite:
    """
    Just bet 1$ on every game based on who the model thinks is the winner
    """

    def __call__(self, model, dataset):
        games, coeffs = dataset.get_games_and_coefficients()
        predictions = model.predict(games) # ngames
        favorites = (predictions < 0.5).astype(np.int32) # ngames
        bets = np.zeros(coeffs.shape, dtype=np.float32) # nbookmakers x ngames x 2
        bets[:, np.arange(games.shape[0]), favorites] = 1

        return bets

    def description(self):
        return f'Every game, bet on model predicted winner'

class UniformBetStrategy:
    """Every game bet 1$ on both players"""

    def __call__(self, model, dataset):
        _, coeffs = dataset.get_games_and_coefficients()
        return np.ones_like(coeffs)

    def description(self):
        return f'Every game, bet 1$ on both'

class NoiseStrategy:
    """Every game bet random value for each player"""

    def __call__(self, model, dataset):
        _, coeffs = dataset.get_games_and_coefficients()
        return np.random.rand(*coeffs.shape)

    def description(self):
        return f'Every game bet random value for each player'

class EveryGameRandomPickStrategy:
    """Every game pick one of the players and bet 1$ on them."""

    def __call__(self, model, dataset):
        games, coeffs = dataset.get_games_and_coefficients()
        picks = (np.random.rand(games.shape[0]) > 0.5) * 1
        bets = np.zeros_like(coeffs)
        bets[:, np.arange(games.shape[0]), picks] = 1
        return bets

    def description(self):
        return f'Every game, bet 1$ on random player'
