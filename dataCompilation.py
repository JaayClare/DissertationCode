import pandas as pd
import os
import weatheranalysis as we
import numpy as np

from sklearn.preprocessing import LabelEncoder

def read_csv(filepath: str) -> pd.DataFrame:
    '''Used in the main loop to read individual csv files for each airline'''
    df = pd.read_csv(filepath)
    return df


def create_master_dataframe(dataframe: list) -> pd.DataFrame:
    '''Accepts list of dataframes and merges into one main dataframe'''
    master_df = pd.concat(dataframe)
    master_df.sort_values(by=['day', 'scheduled_departure'], inplace=True)
    master_df.reset_index(drop=True, inplace=True)
    return master_df


def label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    dest_encoder = LabelEncoder()
    tail_encoder = LabelEncoder()
    df['dest_airport'] = dest_encoder.fit_transform(df['dest_airport'])
    df['tail_num'] = tail_encoder.fit_transform(df['tail_num'])
    return df


def add_weather_to_dataframe(dataframe: pd.DataFrame, wind: dict, cumulative: pd.DataFrame) -> pd.DataFrame:
    '''Adds airfield wind direction and regional weather data for master dataframe'''
    dataframe['airfield_wind_dir'] = dataframe['day'].apply(lambda w: wind[w])

    columns_to_add = list(cumulative.columns)

    for i in range(len(dataframe)):
        current_day = df.loc[i, 'day']
        current_day_conditions = cumulative.loc[cumulative.index == current_day].values[0]
        dataframe.loc[i, columns_to_add] = current_day_conditions

    return dataframe




if __name__ == '__main__':
    folder_path = '/Users/jamesclare/Documents/Python/DissertationCode/ProcessedData'

    files = [f for f in os.listdir(folder_path)]
    dataframes = []
    for f in files:
        file = pd.read_csv(os.path.join(folder_path, f))
        dataframes.append(file)

    # Call subprocess from weather extraction program for airfield conditions
    wind_directions = we.boston_logan_specific_wind()

    # Call subprocess from weather extraction program for cumulative conditions
    cumulative_conditions = we.cumulative_weather_data()

    # Create master dataframe by concatenation each airline
    df = create_master_dataframe(dataframes)

    df = label_encoding(df)

    # Add weather to the master dataframe and save as csv
    df = add_weather_to_dataframe(df, wind_directions, cumulative_conditions)

    # Exporting the master file to CSV
    df.to_csv('/Users/jamesclare/Documents/Python/DissertationCode/FullfileEncoded.csv', index=False)
