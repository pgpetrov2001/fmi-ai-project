import numpy as np

def rolling_returns(returns, investment):
    cumulative_investment = investment.cumsum()
    return np.divide(
        returns.cumsum(),
        cumulative_investment,
        out=np.zeros_like(returns),
        where=cumulative_investment!=0
    )
