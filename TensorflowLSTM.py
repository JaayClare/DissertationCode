import pandas as pd
import numpy as np
import random as ra
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM, RNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
filepath = '/Users/jamesclare/Documents/Python/DissertationCode/FullfileEncoded.csv'


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

    # Scale the features using Min/Max Normalization
    scaler = MinMaxScaler()
    X = df[feature_names].values
    X = scaler.fit_transform(X)

    # Extract the Target
    y = df['delay_above_15']

    # Split and Reshape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False, random_state=42)
    X_train = X_train.reshape(len(X_train), 18, 1)
    X_test = X_test.reshape(len(X_test), 18, 1)


    return X_train, X_test, y_train, y_test



def LSTM_Model(X_train, y_train, X_test, y_test):
    # Hyper Parameters
    batch_size = 100
    epochs = 20

    # Model Architecture
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(10, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))



    # Assemble the optimizer with learning rate
    optimizer = Adam(learning_rate=0.0001)

    # Compile model and declare loss functions/accuracy metric
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])


    # Fit the data and run model
    history = model.fit(X_train, y_train,
                        epochs=epochs, batch_size=batch_size,
                        validation_split=0.2, verbose=2)

    model.summary()

    # Evaluation of the model
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
    print('Loss:', loss)
    print('Accuracy: ', accuracy)

    print(history.history['loss'])
    print(history.history['binary_accuracy'])

    print(history.history['val_loss'])
    print(history.history['val_binary_accuracy'])



def main():
    X_train, X_test, y_train, y_test = return_X_y(read_in_datafile(filepath), undersample=False)
    LSTM_Model(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()
