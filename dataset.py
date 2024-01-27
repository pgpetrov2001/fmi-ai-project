import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class Dataset:
    def __init__(self, df):
        self.df = df.copy(deep=True) #TODO: if needed to increase performance, remove unneeded columns in derived class

    def __len__(self):
        return len(self.df)

    def noise(
        self,
        dropout=0.03,
        swap_prob=0.05,
        swap_max_timediff=None,
        swap_max_dist=10
    ):
        #TODO: implement
        self.df = self.df
        return self.clone()


class TennisDataset(Dataset):
    def __init__(self, df, bookmakers=['PS'], test=False, players_index=None):
        """
        bookmakers='B365'|'PS'|'both'

        This dataset class is modeled around this kaggle dataset:
        https://www.kaggle.com/datasets/edouardthomas/atp-matches-dataset?resource=download

        In short it has tennis match info from around 2001 to about 2020,
        and also the corresponding bookmaker closing coefficients for the matches.
        Coefficients are available for Bet365 and Pinnacle Sports.

        """
        super().__init__(df)

        for bookmaker in bookmakers:
            if bookmaker not in ['B365', 'PS']:
                raise Exception(f'Invalid value for bookmaker f{bookmaker}, must be either B365 or PS')

        self.bookmakers = bookmakers[:]
        self.test = test

        if players_index is None:
            players = set(np.concatenate([self.df['Winner'].unique(), self.df['Loser'].unique()]))
            players = list(players)
            self.players_index = pd.Index(players)
        else:
            self.players_index = players_index

        self.winners = self.players_index.get_indexer(df['Winner'])
        self.losers = self.players_index.get_indexer(df['Loser'])

        if test:
            #here we extract information about the data needed for testing/validation purposes
            #to do that, first we need to remove any information about who is the winner or loser from the input data
            mask = np.random.randint(0,2, size=len(df)).astype(bool)
            df = self.df.copy(deep=False)
            df.loc[mask, ['Winner', 'Loser', 'B365W', 'B365L', 'PSW', 'PSL']] = df.loc[mask, ['Loser', 'Winner', 'B365L', 'B365W', 'PSL', 'PSW']]
            self.games_participants = np.zeros((len(df), 2), dtype=np.int32)
            self.games_coefficients = np.zeros((len(self.bookmakers), len(df), 2), dtype=np.float32)
            self.games_participants[:,0] = self.players_index.get_indexer(df['Winner'])
            self.games_participants[:,1] = self.players_index.get_indexer(df['Loser'])

            for idx, bookmaker in enumerate(self.bookmakers):
                self.games_coefficients[idx,:,0] = df[f'{bookmaker}W']
                self.games_coefficients[idx,:,1] = df[f'{bookmaker}L']

    def clone(self):
        return TennisDataset(self.df, test=self.test, bookmakers=self.bookmakers[:], players_index=self.players_index.copy())

    def take(self, start_date, end_date, **kwargs):
        # in this dataset dates are in ISO format (yyyy-mm-dd)
        filtered_df = self.df[(self.df['Date'] >= start_date.isoformat()) * (self.df['Date'] <= end_date.isoformat())]
        return TennisDataset(filtered_df, bookmakers=self.bookmakers[:], **kwargs)

    def get_games_and_coefficients(self):
        return self.games_participants, self.games_coefficients

    def get_winners(self):
        return self.winners

    def winnings(self, bets, model):
        bets = np.array(bets).reshape(self.games_coefficients.shape)
        mask = self.games_participants == self.winners[:, np.newaxis]
        assert (mask.sum(axis=-1) == np.ones(self.games_participants.shape[0])).all() # only 1 wins

        revenue = (~np.isnan(self.games_coefficients)) * (bets * np.nan_to_num(self.games_coefficients-1) * mask - bets * (~mask))
        cumulative_revenue = revenue.sum(axis=0).sum(axis=-1).cumsum()
        cumulative_bets    = bets   .sum(axis=0).sum(axis=-1).cumsum()
        roi_winnings = np.divide(
            cumulative_revenue,
            cumulative_bets,
            out=np.zeros_like(cumulative_revenue),
            where=cumulative_bets!=0
        )
        return roi_winnings

    def winLoseDataset(self):
        table = np.zeros((len(self.df), len(self.players_index)))
        table[np.arange(table.shape[0]), self.winners] = 1
        table[np.arange(table.shape[0]), self.losers] = -1
        target = np.ones(table.shape[0])

        return table, target
