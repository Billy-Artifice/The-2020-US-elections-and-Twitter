import geopandas
from shapely.geometry import Point, Polygon
import pandas as pd
import matplotlib.pyplot as pyplot

def visualize_geo_tweet(tweet_df,authority_code,world):
    points = [Point(x_y_coordinate) for x_y_coordinate in zip(tweet_df['long'], tweet_df['lat'])]
    geo_data_frame = geopandas.GeoDataFrame(tweet_df, crs=authority_code, geometry=points)
    figure, axes_object = pyplot.subplots(1)
    world.plot(ax=axes_object, edgecolors='black')
    geo_data_frame.plot(ax=axes_object, color='red', marker='x', markersize=2)
    axes_object.axis('off')
    pyplot.show()


trump = pd.read_csv('./donaldtrump_clean.csv')
biden = pd.read_csv('./joebiden_clean.csv')
tweet_data = pd.concat([trump, biden])
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
tweet_data = pd.concat([trump, biden]).dropna()
visualize_geo_tweet(tweet_data,{'init': 'EPSG:4326'},world)