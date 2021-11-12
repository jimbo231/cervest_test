import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
from python.functions import open_net_cdf_file, plot_coordinates, get_variable_values, create_grid_charts, clean_anomalies


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
create_grid_charts(weather_data, save_path='charts/anomaly_chart.png', clean_anomalies=False)
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
master_df['group'] = master_df['longitude'].astype(str) + master_df['latitude'].astype(str)
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
        # then we have the start of the heatwave
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


master_df['unique_heatwave'] = master_df.apply(get_longest_heatwave, axis=1)
check = master_df.loc[master_df['heatwave'] == 1]

# now we just need the sum of heatwaves by

master_df['heatwave'].groupby(master_df['test']).sum



data = {'A':[1,4,3,5],'B':[0,6,3,0],'C':[1,1,3,0]} #sample data
df = pd.DataFrame(data)

for rindex, row in df.iterrows():
        print("row", rindex, "column ", cindex, "value ", value)






master_df['cumsum_heatwave'] = master_df['heatwave'].groupby(master_df['group'], master_df['year']).transform(lambda x: x.cumsum())

    master_df.groupby('group', 'year')['heatwave'].transform(lambda x: x.cumsum)



master_df.columns

master_df.groupby('group', 'year')['heatwave'].cumsum

# basically an if else statement, if 1 in heatwave and shift value is zero then 1,
# elif 1 in heatwave and shift value = 1 (in a heatwave) therefore cumsum

master_df.columns

master_df.VAR_2T.quantile()




















