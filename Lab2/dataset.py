import matplotlib.pyplot as plt
import pandas as pd


def read(path='../seattleWeather_1948-2017.csv'):

    dataset = pd.read_csv(path)

    dataset['TMEAN'] = ( dataset['TMAX'] + dataset['TMIN'] ) / 2
    dataset['PRCP_YESTERDAY'] = dataset['PRCP'].shift(1)

    dataset = dataset[dataset['PRCP_YESTERDAY'].notnull()]
    dataset = dataset[dataset['RAIN'].notnull()]

    dataset = dataset.sort_values(['RAIN'])

    dataset_rain_true = dataset[dataset['RAIN'] == True]
    dataset_rain_false = dataset[dataset['RAIN'] == False]

    result = [
        dataset.as_matrix(['PRCP_YESTERDAY', 'TMEAN', 'RAIN']),
        dataset_rain_true.as_matrix(['PRCP_YESTERDAY', 'TMEAN', 'RAIN']),
        dataset_rain_false.as_matrix(['PRCP_YESTERDAY', 'TMEAN', 'RAIN']),
    ]

    return result


def plot(rain_true, rain_false):

    # rain_true = ['RAIN_YESTERDAY', 'TMEAN']
    # rain_false = ['RAIN_YESTERDAY', 'TMEAN']

    figure = plt.figure()

    plt.plot(rain_false[:, 0], rain_false[:, 1], 'o', mfc='none', label='Rainy')
    plt.plot(rain_true[:, 0], rain_true[:, 1], 'k.', label='Dry')

    plt.xlabel('$PRCP_{yesterday} INCH$')
    plt.ylabel('$TMEAN_{yesterday}$')
    plt.show()

    return figure
