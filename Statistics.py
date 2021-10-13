import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt
plt.gcf().subplots_adjust(bottom=0.25)


df = pd.read_csv('/Users/jamesclare/Documents/Python/DissertationCode/Fullfile.csv')
names = ['Alaska', 'American', 'Delta', 'Frontier', 'Hawaiian', 'JetBlue', 'SouthWest', 'Spirit']
carrier_dict = dict(zip(range(len(names)), names))
df['carrier'] = df['carrier'].apply(lambda c: carrier_dict[c])


def number_of_flights_per_airline(df):
    graph = sns.countplot(data=df, x='carrier')
    graph.set_xticklabels(names)
    plt.xticks(rotation=45)
    plt.ylabel('Number of Flights')
    plt.title('Number of Flights Per Airline')
    plt.xlabel('Airline')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/FlightsPerAirline.png')



def destinations_reached():
    destinations = df['dest_airport'].value_counts()
    fig, ax = plt.subplots(figsize=(10,18))
    sns.barplot(ax=ax, y=destinations.index, x=destinations.values, edgecolor='white')
    plt.ylabel('Airport Code')
    plt.title('Destinations reached from Boston Logan')
    plt.xlabel('Frequency')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/Destinations.png')



def delay_distribution(df):
    df = df.loc[(df.total_delay > 1) & (df.total_delay < 500)]
    sns.histplot(data=df, x='total_delay')
    plt.ylabel('Frequency')
    plt.title('Delay Historgram for flights with delay of 1 < delay < 500')
    plt.xlabel('Delay Category (minutes)')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/DelayHistogram.png')


def total_delay_distribution_with_carrier(df):
    df = df.loc[(df.total_delay >= 1) & (df.total_delay <= 180)]
    sns.histplot(data=df, x='total_delay', hue='carrier')
    plt.ylabel('Frequency')
    plt.title('Delay Histogram for flights with delay of 1 <= delay <= 180')
    plt.xlabel('Category')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/DelayHistogramCarrier3hour.png')



def delay_distribution_per_carrier(df):
    df = df.loc[(df.total_delay >= 1) & (df.total_delay <= 60)]
    sns.violinplot(data=df, x='carrier', y='total_delay')
    plt.title('Delay Distribution per Carrier')
    plt.xlabel('Carrier')
    plt.ylabel('Count')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/DelayDistributionPerCarrier.png')



def time_buckets(df):
    df['regular_time'] = pd.to_datetime(df['scheduled_departure'], unit='m')
    df['hour'] = df['regular_time'].apply(lambda t : t.hour)
    grouped = df.groupby('hour')['total_delay'].mean()
    sns.barplot(x=grouped.index, y=grouped.values)
    plt.title('Mean Delay Per Hour')
    plt.xlabel('Hour')
    plt.ylabel('Mean Delay')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/MeanDelayPerHour.png')


def movements_per_hour(df):
    df['regular_time'] = pd.to_datetime(df['scheduled_departure'], unit='m')
    df['hour'] = df['regular_time'].apply(lambda t : t.hour)
    grouped = df.groupby('hour')['flight_number'].count()
    print(grouped)
    sns.lineplot(x=grouped.index, y=grouped.values)
    plt.title('Cumulative Movements Per Hour over monthly period')
    plt.xlabel('Hour')
    plt.ylabel('Movements')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/MovementsPerHour.png')


def total_monthly_delay_per_carrier(df):
    group = df.groupby('carrier').agg({'flight_number' : 'count',
                                       'total_delay' : 'sum'})

    group['ratio'] = group['total_delay'] / group['flight_number']
    sns.barplot(data=group, x=group.index, y='ratio')
    plt.title('Cumulative flight delay divided by total number of flights')
    plt.xlabel('Carrier')
    plt.ylabel('Ratio')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/DelayRatio.png')


def mean_flight_per_carrier(df):
    data = df.groupby('carrier')['total_scheduled_time'].mean()
    sns.barplot(x=data.index, y=data.values)
    plt.title('Mean Scheduled Journey Time')
    plt.xlabel('Carrier')
    plt.ylabel('Time (minutes)')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/MeanMovementTime.png')


def mean_delay_per_carrier(df):
    group = df.groupby('carrier')['total_delay'].mean()
    print(group)


def delay_summary_per_carrier(df):
    group = df.groupby('carrier')['total_delay'].describe()
    print(group)



####### Weather Section ########


def rain_wind_delay_compare_per_day(df):
    df = df.replace(np.nan, 0)
    grouped = df.groupby('day').agg({'rainfall' : 'mean',
                                    'average_wind' : 'mean',
                                    'late_aircraft_delay' : 'mean'}).sort_values(by='day')

    sns.lineplot(data=grouped)
    plt.title('Monthly Rainfall, wind and late aircraft delay')
    plt.xlabel('Day of Month')
    plt.ylabel('Value')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/RainfallCompare.png')



def binary_event_days(df):
    grouped = df.groupby('day').agg({'smoke_haze' : 'mean',
                                    'thunder' : 'mean',
                                    'fog' : 'mean',
                                    'high_wind' : 'mean',
                                    'total_delay': 'mean'}).sort_values(by='day')

    grouped['total_delay'] = grouped['total_delay'] / 10
    sns.lineplot(data=grouped)
    plt.title('binary weather events & late aircraft delay')
    plt.xlabel('Day of Month')
    plt.ylabel('Value')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/BinaryWeather.png')
    plt.clf()

    sns.heatmap(grouped.corr(), annot=True)
    plt.title('binary weather events & late aircraft Heatmap')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/Heatmap.png')



def wind_reading(df):
    windy_days = sorted(df['average_wind'].unique(), reverse=True)[:10]
    non_windy_days = sorted(df['average_wind'].unique(), reverse=True)[-10:]

    windy_flights = df.loc[df['average_wind'].isin(windy_days)].groupby('day')['total_delay'].mean()
    windy_flights.reset_index(inplace=True, drop=True)

    non_windy_flights = df.loc[df['average_wind'].isin(non_windy_days)].groupby('day')['total_delay'].mean()
    non_windy_flights.reset_index(inplace=True, drop=True)

    sns.barplot(x=windy_flights.index, y=windy_flights.values, color='blue')
    sns.barplot(x=non_windy_flights.index, y=non_windy_flights.values, color='red')

    plt.title('Top/Bottom 10 windy Days vs mean delay')
    plt.xlabel('Least 10 Windy Days in Red\nMost 10 Windy days in Blue')
    plt.ylabel('Delay')
    plt.savefig('/Users/jamesclare/Documents/Python/DissertationCode/Figures/Top10Windy.png')



def wind_heading_delay(df):
    grouped = df.groupby('airfield_wind_dir')['total_delay'].mean()
    print(grouped)




##### Functions Calls for Air Movement Graphs

number_of_flights_per_airline(df)
destinations_reached(df)
delay_distribution(df)
total_delay_distribution_with_carrier(df)
delay_distribution_per_carrier(df)
delay_distribution_per_destination(df)
time_buckets(df)
movements_per_hour(df)
total_monthly_delay_per_carrier(df)
mean_flight_per_carrier(df)

### Statistic Function Calls for Air Movement
mean_delay_per_carrier(df)
delay_summary_per_carrier(df)


#### Functions Calls for Weather Graphs
rain_wind_delay_compare_per_day(df)
binary_event_days(df)
wind_reading(df)
wind_heading_delay(df)
