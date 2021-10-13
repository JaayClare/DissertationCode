import pandas as pd
import numpy as np
import datetime as dt
import os


def read_and_rename(filepath: str) -> pd.DataFrame:
    '''Reads in raw CSV file and renames columns'''
    df = pd.read_csv(filepath, skiprows=7)
    df.rename(columns={'Date (MM/DD/YYYY)' : 'date',
                       'Carrier Code' : 'carrier',
                       'Flight Number' : 'flight_number',
                       'Tail Number': 'tail_num',
                       'Destination Airport' : 'dest_airport',
                       'Scheduled departure time' : 'scheduled_departure',
                       'Actual departure time' : 'actual_departure',
                       'Scheduled elapsed time (Minutes)' : 'total_scheduled_time',
                       'Actual elapsed time (Minutes)' : 'actual_scheduled_time',
                       'Departure delay (Minutes)' : 'departure_delay',
                       'Wheels-off time' : 'wheels_off',
                       'Taxi-Out time (Minutes)' :  'taxi_time',
                       'Delay Carrier (Minutes)' : 'carrier_delay',
                       'Delay Weather (Minutes)' : 'weather_delay',
                       'Delay National Aviation System (Minutes)' : 'nas_delay',
                       'Delay Security (Minutes)' : 'security_delay',
                       'Delay Late Aircraft Arrival (Minutes)' : 'late_aircraft_delay'}, inplace=True)

    df.drop(df.tail(1).index, inplace=True)

    return df


def process_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''Applies processing to date and carrier columns'''
    carriers = ['AS', 'AA', 'DL', 'F9', 'HA', 'B6', 'WN', 'NK']
    carrier_dict = dict(zip(carriers, range(len(carriers))))

    df['carrier'] = df['carrier'].apply(lambda c: carrier_dict[c])
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df.drop(columns='date', inplace=True)


    delay_columns = ['carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
    df['total_delay'] = df[delay_columns].sum(axis=1)
    df['delay_above_15'] = np.where(df['total_delay'] >= 15, 1, 0)

    return df


def time_to_minutes(time_):
    '''Used as a subprocess by time conversions function'''
    hour_component = 60 * int(time_.split(':')[0])
    minute_component = int(time_.split(':')[1])

    return hour_component + minute_component


def time_conversions(df: pd.DataFrame) -> pd.DataFrame:
    '''Converts time (HH:MM) into minute component only'''
    df['scheduled_departure'] = df['scheduled_departure'].apply(time_to_minutes)
    df['actual_departure'] = df['actual_departure'].apply(time_to_minutes)
    df['wheels_off'] = df['wheels_off'].apply(time_to_minutes)

    return df


def main_process(path, save_path):
    data = read_and_rename(path)
    data = time_conversions(process_columns(data))
    filename = os.path.join(save_path, path.split('/')[-1][:-4]) + 'processed.csv'
    data.to_csv(filename, index=False)


if __name__ == '__main__':
    source_folder_path = '/Users/jamesclare/Documents/Python/DissertationCode/Data'
    processed_folder_path = '/Users/jamesclare/Documents/Python/DissertationCode/ProcessedData'

    files = [f for f in os.listdir(source_folder_path)]
    for f in files:
        main_process(os.path.join(source_folder_path, f), processed_folder_path)
