from datetime import date, timedelta

import numpy as np
import pandas as pd

from dataset import TennisDataset
from models import LinearRegressionModel
from evaluate import evaluate_strategy
from strategies import *

if __name__ == '__main__':
    df = pd.read_csv('atp_data.csv')
    dataset = TennisDataset(df)
    winnings = evaluate_strategy(
        dataset,
        LinearRegressionModel,
        expected_value_cutoff,
        min_date=date(year=2005, month=1, day=1),
        max_date=date(year=2018, month=3, day=4),
        test_period=timedelta(days=31*4-3),
        plot=True,
        verbose=False,
    )
