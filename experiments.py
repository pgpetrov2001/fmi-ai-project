import sys
import pandas as pd
import numpy as np

def get_data_from_df(df, players_index):
    table = np.zeros((len(df), len(players)))
    table[np.arange(table.shape[0]), players_index.get_indexer(df['Winner'])] = 1
    table[np.arange(table.shape[0]), players_index.get_indexer(df['Loser'])] = -1
    target = np.ones(table.shape[0])

    #signs = np.random.randint(0, 2, target.shape[0])*2 - 1
    #table *= signs[:,np.newaxis]
    #target *= signs
    return table, target

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_ratings(table, target):
    C = table.T@table
    b = table.T@target

    ratings = np.linalg.solve(C, b)
    return ratings

def get_loss(table, ratings, target):
    return np.linalg.norm(table@ratings - target)

def get_accuracy(table, ratings, target):
    return (np.sign(table@ratings) == np.sign(target)).sum() / target.shape[0]

def test_betting_strategy(df, ratings, players_index, bookmaker='B365'):
    winners = players_index.get_indexer(df['Winner'])
    losers = players_index.get_indexer(df['Loser'])
    p_winner = sigmoid(ratings[winners] - ratings[losers])

    bookmaker_winnner_coeff = df[f'{bookmaker}W']
    bookmaker_loser_coeff   = df[f'{bookmaker}L']

    #some percentage of coefficients are missing:
    #print(bookmaker_winnner_coeff.isna().sum() / len(bookmaker_winnner_coeff))
    #print(bookmaker_loser_coeff.isna().sum() / len(bookmaker_loser_coeff))

    # every time when the expected value of return on investment is above 10%, according to our probability model
    # our strategy puts in a bet of 1$ 

    # sanity check: if the coefficient for one player is missing, the coefficient for the other is also missing
    #assert ((bookmaker_winnner_coeff.isna() * bookmaker_loser_coeff.isna()) == bookmaker_winnner_coeff.isna()).all()
    #apparently not always true

    expected_value_winner = bookmaker_winnner_coeff * p_winner     - (1-p_winner)
    expected_value_loser  = bookmaker_loser_coeff   * (1-p_winner) - p_winner
    bet_winner_mask = (~bookmaker_winnner_coeff.isna()) * (p_winner > 1/2) * (expected_value_winner > 1.05)
    bet_loser_mask  = (~bookmaker_loser_coeff.isna())   * (p_winner < 1/2) * (expected_value_loser  > 1.05)

    #this is the amount of money we won:
    bet_winner = (bookmaker_winnner_coeff[bet_winner_mask] - 1).sum()
    #this is the amount of money we lost:
    bet_loser = bet_loser_mask.sum()
    bet_overall = (bet_winner_mask | bet_loser_mask).sum()
    print('bets put')
    print(bet_overall)

    return bet_overall, bet_winner, bet_loser

def test_arbitrage_strategy(df, bookmaker='B365'):
    winner_coeff = df[f'{bookmaker}W']
    loser_coeff = df[f'{bookmaker}L']
    coeff_present = ~winner_coeff.isna()

    return coeff_present.sum(), (winner_coeff-1).sum(), coeff_present.sum()

def show_betting_strategy_results(bets_put, bets_won, bets_lost, verbose=False):
    if bets_put == 0:
        print('no bets put at all')
        return

    # print('bets put: ', bets_put)
    if verbose:
        print('bets won: ', bets_won)
        print('bets lost: ', bets_lost)
        print('earnings: ', bets_won - bets_lost)
    print(f'return on investment: {100 * (bets_won - bets_lost) / bets_put:.2f}%')

df = pd.read_csv('atp_data.csv')

for start_year in range(2005, 2017):
    train_start_date = f'{start_year}-01-01'
    train_end_date = f'{start_year+2}-01-01'
    # print('******************************************************************')
    # print(f'Training in range {train_start_date} to {train_end_date}')
    train = df[(df['Date'] >= train_start_date) * (df['Date'] < train_end_date)]

    players = set(np.concatenate([train['Winner'].unique(), train['Loser'].unique()]))
    players = list(players)
    players_index = pd.Index(players)

    train_table, train_target = get_data_from_df(train, players_index)

    ratings = get_ratings(train_table, train_target)

    # print('Reference loss: ', get_loss(train_table, np.random.rand(train_table.shape[1]), train_target))
    #print('Reference accurracy: ', get_accuracy(train_table, np.random.rand(train_table.shape[1]), train_target))
    # print('Loss: ', get_loss(train_table, ratings, train_target))
    # print(f'Train accuracy:  {100*get_accuracy(train_table, ratings, train_target):.2f}%')
    # print('################################################################################')

    for end_month in range(5, 6):
        start_date = f'{start_year+2}-01-01'
        end_date = f'{start_year+2}-{end_month:0>2}-01'

        test = df[(df['Date'] >= start_date) * (df['Date'] <= end_date)]
        print('games total', len(test))
        games_with_known_players_mask = (players_index.get_indexer(test['Winner']) != -1) * (players_index.get_indexer(test['Loser']) != -1)
        test = test[games_with_known_players_mask]
        print('games with known players', len(test))
        test_table, test_target = get_data_from_df(test, players_index)

        # print('---------------------------------------------------------------------')
        print(f'{start_date} - {end_date}')
        # print(f'game outcome prediction accuracy: {100 * get_accuracy(test_table, ratings, test_target):.2f}%')

        for bookmaker in ['PS']:
            bets_put, bets_won, bets_lost = test_betting_strategy(test, ratings, players_index, bookmaker)
            show_betting_strategy_results(bets_put, bets_won, bets_lost)
            #print(f'mindless arbitrage strategy with {bookmaker}:')
            #show_betting_strategy_results(*test_arbitrage_strategy(test, bookmaker))


    #print(df.apply(lambda col: col.describe()))
