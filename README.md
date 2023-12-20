This is a neural network project aimed at predicting the outcomes of NBA matches and optimizing said predictions.

# Preprocessing data:
The NBA game data are from https://www.kaggle.com/datasets/nathanlauga/nba-games?resource=download. Within the data, the games.csv file was chosen for analytics. After converting the data to a pandas dataframe, the average values for each team's statistics were taken and added to a new dataframe. Because averages couldn't be taken on the first game a team played, the first games of the season were dropped. A second loop was run to add a margin feature which improved accuracy drastically, raising it from 54% to 65%.

# Neural Network:
The preprocessed data was split into a fourth being used for testing and the rest used for predictions. With two layers, binary entropy loss function, and Adam optimization, the current neural network accuracy is 65%.


# Future:
The accuracy of the latest model, MLPWithMargins, in combination with Preprocessing.py have yielded an accuracy of 65%. The goal for the foreseeable future is to increase this accuracy by experimenting with hyperparameters, LSTM and RNN models, and other features.
