import numpy as np
import pandas as pd
import random as ra
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold


from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier



def read_in_datafile(path: str) -> pd.DataFrame:
    '''Read in CSV file and return DataFrame'''
    df = pd.read_csv(path)



    return df


def return_X_y(df: pd.DataFrame, undersample=False) -> tuple:
    '''Returns features and labels from dataset'''
    feature_names = ['carrier', 'flight_number', 'tail_num', 'dest_airport',
                    'scheduled_departure', 'total_scheduled_time', 'taxi_time', 'day',
                    'airfield_wind_dir', 'thunder', 'smoke_haze', 'high_wind',
                    'average_wind', 'fog', 'min_temp', 'max_temp',
                    'total_sun', 'rainfall']

    if undersample:
        negative_indices = df.loc[df.delay_above_15 == 0].index
        indices = ra.choices(negative_indices, k=3000)
        df = df.drop(df.index[indices])

    X = df[feature_names].values
    y = df['delay_above_15'].values

    return X,y



def perform_mlp(X: np.array, y: np.array, folds: int):
    '''Runs the Multi-Layer Perceptron with k-fold cross validation'''

    mlp = MLPClassifier(random_state=1, max_iter=300, solver='adam')
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    mlp_scores = cross_val_score(mlp, X, y, cv=skf)

    mlp_accuracy_score = mlp_scores.mean()
    mlp_accuracy_st_dev = mlp_scores.std()

    print('All MLP Scores:', mlp_scores)
    print('MLP Accuracy Score: {:.2f}'.format(mlp_accuracy_score))
    print('MLP Standard Deviation: {:.2f}'.format(mlp_accuracy_st_dev))

    return dict(enumerate(mlp_scores))


def perform_random_forest(X: np.array, y: np.array, folds: int):
    '''Runs the Random Forest Classifier with k-fold cross validation'''

    rfc = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=0)
    skf = StratifiedKFold(n_splits=folds, shuffle=True)

    rfc_scores = cross_val_score(rfc, X, y, cv=skf)

    rfc_accuracy_score = rfc_scores.mean()
    rfc_accuracy_st_dev = rfc_scores.std()

    print('All RFC Scores:', rfc_scores)
    print('RFC Accuracy Score: {:.2f}'.format(rfc_accuracy_score))
    print('RFC Standard Deviation: {:.2f}\n'.format(rfc_accuracy_st_dev))

    return dict(enumerate(rfc_scores))



def perform_adaboost(X: np.array, y: np.array, folds: int):
    '''Runs the AdaBoost Classifier with k-fold cross validation'''

    ada = AdaBoostClassifier(n_estimators=400, learning_rate=1, random_state=0)
    skf = StratifiedKFold(n_splits=folds, shuffle=True)
    ada_scores = cross_val_score(ada, X, y, cv=skf)

    ada_accuracy_score = ada_scores.mean()
    ada_accuracy_st_dev = ada_scores.std()

    print('All ADA Scores: ', ada_scores)
    print('ADA Accuracy Score: {:.2f}'.format(ada_accuracy_score))
    print('ADA Standard Deviation: {:.2f}\n'.format(ada_accuracy_st_dev))

    return dict(enumerate(ada_scores))



def plot_score(scores: dict, title: str) -> None:
    sns.lineplot(x=scores.keys(), y=scores.values())
    plt.xticks(range(5), ['Fold: ' + str(i) for i in range(1,6)])
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/{}'.format(title))
    plt.clf()


def main():
    '''Run Main Process'''
    path = '/Users/jamesclare/Documents/Python/DissertationCode/FullfileEncoded.csv'
    data = read_in_datafile(path)
    X,y = return_X_y(data, undersample=False)

    score_for_mlp = perform_mlp(X, y, 5)
    score_for_rfc = perform_random_forest(X, y, 5)
    score_for_ada = perform_adaboost(X, y, 5)

    plot_score(score_for_mlp, 'Multi Layer Perceptron Undersampled')
    plot_score(score_for_rfc, 'Random Forest Classifier Undersampled')
    plot_score(score_for_ada, 'AdaBoost Classifier Undersampled')





if __name__ == '__main__':
    main()
