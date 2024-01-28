import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from dataset import TennisDataset
from models import LinearRegressionModel
from evaluate import evaluate_strategy
from strategies import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser("""
    Evaluate and compare different betting strategies using different prediction models.
    When using LinearRegressionModel if the error "numpy.linalg.LinAlgError: Singular matrix"
    occurrs, just rerun the program with the exact same arguments.
    """)
    parser.add_argument('--save-plot', '-s', type=str, help='Where to save the generated plot. If not specified, will not be saved.')
    args = parser.parse_args()

    df = pd.read_csv('data/atp_data.csv')
    dataset = TennisDataset(df, bookmakers=['B365', 'PS'])
    evaluate_strategy(
        dataset,
        LinearRegressionModel,
        ExpectedValueCutoff(threshold=0.15),
        min_date=date(year=2005, month=1, day=1),
        max_date=date(year=2018, month=3, day=4),
        test_period=timedelta(days=31*4-3),
        plot=True,
        plot_x='bets_placed',
        plot_y='rolling_roi',
        save_plot=args.save_plot,
        verbose=False,
    )
