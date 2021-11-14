import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from python.functions import open_net_cdf_file, plot_coordinates, get_variable_values, create_grid_charts
from statsmodels.tsa.seasonal import seasonal_decompose
import pymannkendall as mk
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


# load the data set
weather_data = open_net_cdf_file('C:/Users/james/Downloads/raw_temp_data_london.nc')

# check min max values for the variables
get_variable_values(weather_data, 'time')
get_variable_values(weather_data, 'latitude')
get_variable_values(weather_data, 'longitude')

# check what the metadata looks like and look into any of the links
for item in weather_data.attrs:
    print(weather_data.attrs[item])

# plot the coordinates we were given to see the locations
uk_map = plot_coordinates(weather_data['VAR_2T']['longitude'].values, weather_data['VAR_2T']['latitude'].values)
uk_map.get_figure().savefig("charts/uk_coordinate_map.png")

# plot each location's temperature over time
create_grid_charts(weather_data, save_path='charts/anomaly_chart.png',
                   clean_anomalies=False, moving_average=False)
mpl.rc('lines', linewidth=0.5)  # change the global line width for all charts
create_grid_charts(weather_data, save_path='charts/anomaly_chart_cleaned.png',
                   clean_anomalies=True, moving_average=False)
create_grid_charts(weather_data, save_path='charts/anomaly_chart_cleaned_ma.png',
                   clean_anomalies=True, moving_average=True)

# calculating the annual longest heatwave length by year


# 1.) first create a master data frame which every location, time and temperature in
def get_master_df(weather_data):

    for i in range(0, 13):
        chart_data = weather_data['VAR_2T'].sel(latitude=weather_data['VAR_2T']['latitude'].values[i],
                                                longitude=weather_data['VAR_2T']['longitude'].values[i])
        chart_df = chart_data.to_dataframe()
        # clean the anomalies in each one
        chart_df['VAR_2T'] = np.where(chart_df['VAR_2T'] > 2000, chart_df['VAR_2T'] / 10, chart_df['VAR_2T'])
        # append it to the master data frame
        if i == 0:
            master_df_return = chart_df
        else:
            master_df_return = master_df_return.append(chart_df)

    return master_df_return


master_df = get_master_df(weather_data)
# 2.) get the 95th percentile as a column by each location
master_df.reset_index(level=0, inplace=True)
master_df['group'] = master_df['longitude'].astype(str) + ',' + master_df['latitude'].astype(str)
master_df['95th'] = master_df['VAR_2T'].groupby(master_df['group']).transform(lambda x: x.drop_duplicates().quantile(0.95))

# 3.) make a heatwave dummy
master_df['heatwave'] = np.where(master_df['VAR_2T'] > master_df['95th'], 1, 0)
master_df['time'] = pd.to_datetime(master_df['time'])
master_df['year'] = pd.DatetimeIndex(master_df['time']).year.astype(str)

# 4.) we want the annual longest heatwave by year and location
master_df['heatwave_shift'] = master_df['heatwave'].groupby(master_df['group']).shift(-1)
unique_heatwave = 1  # need this to make the new unique heatwave reference each time


def get_longest_heatwave(row):
    global unique_heatwave
    if row['heatwave'] == 0 and row['heatwave_shift'] == 1:
        # then we are just before the start of the heatwave
        val = 0
    elif row['heatwave'] == 1 and row['heatwave_shift'] == 1:
        # then we are in the heatwave
        val = unique_heatwave
    elif row['heatwave'] == 1 and row['heatwave_shift'] == 0:
        # then we are at the end of the heatwave
        val = unique_heatwave
        unique_heatwave += 1
    else:
        val = 0
    return val


# create a unique heatwave indicator for each one
master_df['unique_heatwave'] = master_df.apply(get_longest_heatwave, axis=1)
master_df['unique_heatwave'] = master_df['unique_heatwave'].astype(str)
# get the sum of the days in each unique heatwave
master_df['heatwave_days'] = master_df['heatwave'].groupby(master_df['unique_heatwave']).transform('sum')
# lastly create an identifier to get the maximum heatwave by year and location
master_df['max_heatwave'] = master_df.groupby(['year', 'group'])['heatwave_days'].transform('max')
master_df['max_heatwave_indicator'] = np.where(master_df['max_heatwave'] == master_df['heatwave_days'], 1, 0)
check = master_df.loc[master_df['heatwave'] == 1]  # quick check

np.where(master_df['VAR_2T'] > master_df['95th'], 1, 0)

# 5.) Create charts for the longest heatwave by each year and location
to_chart = master_df[master_df['max_heatwave_indicator'] == 1].groupby(['year', 'group'], as_index=False).agg({'heatwave_days':'max'})
to_chart['heatwave_days'] = np.log(to_chart['heatwave_days'])


def get_current_heatwave_df(df, list_to_pick, number):
    current_chart = df[df['group'] == list_to_pick[number]]
    current_chart['year'] = pd.to_datetime(current_chart['year'])
    current_chart.set_index('year', inplace=True)
    current_chart.drop('group', axis=1, inplace=True)
    return current_chart


def create_grid_bar_charts(df, save_path):
    fig, axes = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(6, 6))
    fig.text(0.5, 0, 'Year', ha='center')
    fig.text(0.01, 0.5, 'Log Length of Heatwave (hours)', va='center', rotation='vertical')
    list_of_groups = df['group'].unique()
    count = 0
    for row in axes:
        row[0].plot(get_current_heatwave_df(df, list_of_groups, 0 + count*3))
        row[0].set_title(list_of_groups[0 + count*3])
        row[1].plot(get_current_heatwave_df(df, list_of_groups, 1 + count*3))
        row[1].set_title(list_of_groups[1 + count * 3])
        row[2].plot(get_current_heatwave_df(df, list_of_groups, 2 + count*3))
        row[2].set_title(list_of_groups[2 + count * 3])
        count += 1
        row[0].tick_params(axis='x', rotation=90)
        row[1].tick_params(axis='x', rotation=90)
        row[2].tick_params(axis='x', rotation=90)

    plt.subplots_adjust(wspace=0.05, hspace=0.5)
    plt.show()
    plt.savefig(save_path)


create_grid_bar_charts(to_chart, save_path='charts/max_heatwave_by_year.png')

# 6.) Report mean, std.dev and max of the distribution of annual heatwaves
to_report = master_df[master_df['max_heatwave_indicator'] == 1].groupby(['year', 'group'], as_index=False).agg({'heatwave_days':'max'})
to_report['heatwave_days'].mean()  # 24.86
to_report['heatwave_days'].max()   # 259
to_report['heatwave_days'].std()   # 27.07


# 7.) is there a statistically significant upward trend in the length of max heatwave days?
# we want to decompose each time series into the trend and seasonality parts
# Multiplicative Decomposition
to_chart = master_df[master_df['max_heatwave_indicator'] == 1].groupby(['year', 'group'], as_index=False).agg({'heatwave_days':'max'})
list_of_groups = to_chart['group'].unique()
get_trend = to_chart[to_chart['group'] == list_of_groups[10]]
get_trend['year'] = pd.to_datetime(get_trend['year'])
get_trend.set_index('year', inplace=True)
get_trend = get_trend[['heatwave_days']]

# Multiplicative Decomposition
result_mul = seasonal_decompose(get_trend['heatwave_days'], model='multiplicative', period=5)
# Plot
plt.rcParams.update({'figure.figsize': (10, 10)})
result_mul.plot()
plt.savefig('charts/trend_vs_seasonality_decomposition.png')

# Testing if the trend is significant using the mann-kendall-test
result_list = []
for i in range(0, 13):
    # get the list of values we are testing
    to_test = list(to_chart[to_chart['group'] == list_of_groups[i]]['heatwave_days'])
    # run the test
    test_result = mk.original_test(to_test)
    # append the result to a list of dictionaries
    result_list.append({
        'location': list_of_groups[i],
        'trend': test_result.trend,
        'result': test_result.h,
        'p_value': round(test_result.p, 4),
        'slope': round(test_result.slope, 4)
    })

# now can look at the results and easily save them as a png
df_results = pd.DataFrame(result_list)


# 7: co-integration test and VAR Model

# 51.5, 0 is position 6 while (51.75, -1.25) does not exist but 51.75, -0.25 is closest to using this one (position 0)
list_of_groups

y_0 = list(to_chart[to_chart['group'] == list_of_groups[6]]['heatwave_days'])
y_1 = list(to_chart[to_chart['group'] == list_of_groups[0]]['heatwave_days'])

# The null hypothesis is no co-integration. Variables in y0 and y1 are assumed to be integrated of order 1, I(1).
# gives a p-value of 0.034 then we can reject the hypothesis that there is no co-integrating relationship
ts.coint(y_0, y_1)

# set up a data frame for the VAR model
y_0 = to_chart[to_chart['group'] == list_of_groups[6]][['year', 'heatwave_days']]
y_0.columns = ['year', '51.5, 0']
y_1 = to_chart[to_chart['group'] == list_of_groups[0]][['year', 'heatwave_days']]
y_1.columns = ['year', '51.75, -0.25']

var_df = y_0.merge(y_1)
# make the year column the index
var_df['year'] = pd.to_datetime(var_df['year'])
var_df.set_index('year', inplace=True)

# split it into the training and test set
n_obs = 4
df_train, df_test = var_df[0:-n_obs], var_df[-n_obs:]

# check if stationary (using a unit root test)
adfuller(var_df['51.5, 0'], autolag='AIC')
adfuller(var_df['51.75, -0.25'], autolag='AIC')
# The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary.

# make the VAR model and select the max lags
model = VAR(df_train)
x = model.select_order(maxlags=8)
print(x.summary())

# next run the model
model = VAR(df_train)
model_fitted = model.fit(1)
model_fitted.summary()

