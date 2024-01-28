import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class PredictionModel:
    def __init__(self):
        pass

    def predict(self, games, games_features=None):
        """
        Receives a numpy array games with the shape (number_of_games, 2).
        The first value is the first player, the second value is the second player.
        """
        pass


class LinearRegressionModel(PredictionModel):
    """
    The data is transformed to a table with the shape of (number_of_games, number_of_players)
    where for each game if players with indices i and j play in the game with index k, table[k,i] = 1 and table[k,j] = -1
    The rows of this table represent the input vectors for the games.
    This is a linear model so there is one parameter for each player, meaning it can be interpreted as a numerical rating for that player.
    The target is a vector consisting of 1s.
    The model is trained using the standard least-squares method, obtaining a global optimum for the parameter values with respect to L2 loss.
    """
    def __init__(self):
        pass

    def train(self, dataset):
        X, Y = dataset.winLoseDataset()

        ghost_players = np.all(X == 0, axis=0)
        num_players = X.shape[1]
        X = X[:,~ghost_players]

        C = X.T@X
        b = X.T@Y

        self.X, self.Y = X, Y
        self.active_ratings = np.linalg.solve(C, b)
        self.ratings = np.zeros(num_players)
        self.ratings[ghost_players] = float('nan')
        self.ratings[~ghost_players] = self.active_ratings

    def loss(self):
        return np.linalg.norm(self.X@self.active_ratings - self.Y)

    def accuracy(self, dataset=None):
        if dataset is None:
            return (np.sign(self.X@self.active_ratings) == np.sign(self.Y)).sum() / self.Y.shape[0]

        games, _ = dataset.get_games_and_coefficients()
        winners = dataset.get_winners()
        first_truths  = (games[:,0] == winners) * (self.predict(games) > 0.5)
        second_truths = (games[:,1] == winners) * (self.predict(games) < 0.5)
        return np.sum(first_truths + second_truths) / games.shape[0]

    def predict(self, games):
        # if we know nothing about one of the players we predict 0.5
        return sigmoid(np.nan_to_num(self.ratings[games[:,0]] - self.ratings[games[:,1]], nan=0))

    @staticmethod
    def description():
        return f'Linear Regression Model'

