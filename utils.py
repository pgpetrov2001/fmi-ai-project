import numpy as np

def calculate_returns(model, bets):
    games, coeffs = model.get_games_and_coefficients()
    winners = model.get_winners()

    bets = np.array(bets).reshape(coeffs.shape)
    mask = games == winners[:, np.newaxis]
    assert (mask.sum(axis=-1) == np.ones(games.shape[0])).all() # only 1 wins

    result = (~np.isnan(coeffs)) * (bets * np.nan_to_num(coeffs-1) * mask - bets * (~mask))
    return result.sum(axis=0).sum(axis=-1), bets.sum(axis=0).sum(axis=-1)

def rolling_returns(returns, investment):
    cumulative_investment = investment.cumsum()
    return np.divide(
        returns.cumsum(),
        cumulative_investment,
        out=np.zeros_like(returns),
        where=cumulative_investment!=0
    )
