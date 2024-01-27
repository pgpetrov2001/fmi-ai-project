import numpy as np
import sys
import pandas as pd

def expected_value_cutoff(model, dataset):
    """
    There is an allocation of 1$ for each bet.
    If the expected value of the ROI of this bet is above a certain fixed cutoff percentage,
    according to our estimate of the probability of our preferred player winning,
    then we place the bet of 1$ on our preferred player, otherwise we don't.
    """
    games, coeffs = dataset.get_games_and_coefficients()
    predictions = model.predict(games) # ngames
    favorites = (predictions < 0.5).astype(np.int32) # ngames
    c = coeffs[:, np.arange(games.shape[0]), favorites] # nbookmakers x ngames
    predictions[predictions < 0.5] = 1-predictions[predictions < 0.5]
    expected_values = predictions * c - (1-predictions) # nbookmakers x ngames
    mask = (predictions != 0.5) * (~np.isnan(expected_values)) * (expected_values > 1.05) # nbookmakers x ngames
    bets = np.zeros(coeffs.shape, dtype=np.float32) # nbookmakers x ngames x 2

    if mask.sum() == 0:
        # decided to not make any bets (most likely due to missing data)
        return bets

    bets[mask, favorites[mask.any(axis=0)]] = 1
    return bets
