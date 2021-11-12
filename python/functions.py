import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame


def open_net_cdf_file(file_path):
    with xr.open_dataset(file_path) as file_nc:
        loaded_data = file_nc
    return loaded_data


def get_variable_values(array_object, var_name):
    min_value = array_object['VAR_2T'][var_name].values.min()
    max_value = array_object['VAR_2T'][var_name].values.max()
    len_value = array_object['VAR_2T'][var_name].values.shape
    return print(f'\n Variable: {var_name} \n Min value: {min_value} \n Max value: {max_value} \n Length: {len_value}')


def plot_coordinates(lat_values, long_values):
    df_use = pd.DataFrame({'longitude': lat_values, 'latitude': long_values})
    geometry = [Point(xy) for xy in zip(df_use['longitude'], df_use['latitude'])]
    gdf = GeoDataFrame(df_use, geometry=geometry)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    axis_definition = world[(world.name == "United Kingdom")].plot(color='white', edgecolor='black')
    return gdf.plot(ax=axis_definition, marker='o', color='red', markersize=15)


def get_chart_df(number_location, weather_data, clean_anomalies):
    chart_data = weather_data['VAR_2T'].sel(latitude=weather_data['VAR_2T']['latitude'].values[number_location],
                                            longitude=weather_data['VAR_2T']['longitude'].values[number_location])
    chart_df = chart_data.to_dataframe()
    if clean_anomalies == 'True':
        chart_df['VAR_2T'] = np.where(chart_df['VAR_2T'] > 2000, chart_df['VAR_2T']/10, chart_df['VAR_2T'])
        chart_df['MA'] = chart_df.VAR_2T.rolling(window=2160*4).mean()
        return chart_df
    else:
        return chart_df


def create_grid_charts(weather_data, save_path, clean_anomalies, moving_average):
    fig, axes = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(6, 6))
    fig.text(0.5, 0, 'Date', ha='center')
    fig.text(0.01, 0.5, 'Degrees Kelvin', va='center', rotation='vertical')

    count = 0
    for row in axes:
        if not clean_anomalies:
            row[0].plot(get_chart_df(0 + count*3, weather_data, 'False')['VAR_2T'])
            row[1].plot(get_chart_df(1 + count*3, weather_data, 'False')['VAR_2T'])
            row[2].plot(get_chart_df(2 + count*3, weather_data, 'False')['VAR_2T'])
        elif clean_anomalies and not moving_average:
            row[0].plot(get_chart_df(0 + count*3, weather_data, 'True')['VAR_2T'])
            row[1].plot(get_chart_df(1 + count*3, weather_data, 'True')['VAR_2T'])
            row[2].plot(get_chart_df(2 + count*3, weather_data, 'True')['VAR_2T'])
        else:
            row[0].plot(get_chart_df(0 + count*3, weather_data, 'True')['MA'])
            row[1].plot(get_chart_df(1 + count*3, weather_data, 'True')['MA'])
            row[2].plot(get_chart_df(2 + count*3, weather_data, 'True')['MA'])
        count += 1
        row[0].tick_params(axis='x', rotation=90)
        row[1].tick_params(axis='x', rotation=90)
        row[2].tick_params(axis='x', rotation=90)

    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.show()
    plt.savefig(save_path)

