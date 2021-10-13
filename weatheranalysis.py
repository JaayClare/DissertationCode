import pandas as pd
import numpy as np


def read_in_and_process_csv(filepath: str) -> pd.DataFrame:
    '''Reads in the raw data and performs initial processing of file'''
    df = pd.read_csv(filepath)
    df['DATE'] = df['DATE'].apply(lambda d: int(d.split('-')[-1]))
    df.set_index('DATE', drop=True, inplace=True)
    cols_to_include = ['STATION', 'WT03', 'WT08', 'WT11', 'WDF5','AWND', 'WT01', 'TMIN', 'TMAX', 'TSUN', 'PRCP']

    renamed_cols = ['station', 'thunder', 'smoke_haze',
                    'high_wind',
                    'direction_5sec_wind', 'average_wind',
                    'fog', 'min_temp',
                    'max_temp', 'total_sun',
                    'rainfall']

    df = df[cols_to_include]
    df = df.rename(columns=dict(zip(cols_to_include, renamed_cols)))
    df = df.replace(np.nan, 0)

    return df



def retrieve_airfield_wind(df: pd.DataFrame) -> dict:
    '''Returns Dictionary of wind directions for each day of the month'''
    airfield_wind = df.loc[df.station == 'USW00014739']['direction_5sec_wind']
    return airfield_wind.to_dict()


def retrieve_averages(df: pd.DataFrame) -> pd.DataFrame:
    '''Groups data by day and applies functions to grouped data'''
    df = df.drop('direction_5sec_wind', axis=1)
    aggregate_functions = {'thunder' : 'max',
                           'smoke_haze' : 'max',
                           'high_wind': 'max',
                           'average_wind' : 'mean',
                           'fog' : 'max',
                           'min_temp' : 'mean',
                           'max_temp' : 'mean',
                           'total_sun' : 'mean',
                           'rainfall' : 'mean'}


    grouped = df.groupby(df.index).agg(aggregate_functions)
    return grouped


def boston_logan_specific_wind() -> dict:
    '''Calls entire process for airfield wind direction only'''
    dataframe = read_in_and_process_csv('/Users/jamesclare/Documents/Python/DissertationCode/WeatherData/all_weather.csv')
    wind_directions = retrieve_airfield_wind(dataframe)
    return wind_directions


def cumulative_weather_data() -> pd.DataFrame:
    '''Calls entire process for the cumulative weather figures'''
    dataframe = read_in_and_process_csv('/Users/jamesclare/Documents/Python/DissertationCode/WeatherData/all_weather.csv')
    averages = retrieve_averages(dataframe)
    return averages
