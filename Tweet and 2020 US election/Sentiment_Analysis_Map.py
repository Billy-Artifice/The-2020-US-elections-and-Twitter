import geopandas
import pandas as pd
import numpy as np
import folium
from folium import Choropleth
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    text = text.replace('\n', ' ')
    sentiment_strength = analyzer.polarity_scores(text)
    return sentiment_strength['compound']


def panda_aggregate(x):
    # agg(lambda x: (x > 0.05).sum())
    return (x>0.05).sum()

def extract_valid_states(polygones):
    invalid_states =["American Samoa","Puerto Rico","Guam", "District of Columbia", "United States Virgin Islands",
                      "Commonwealth of the Northern Mariana Islands"]
    valid_states = ''
    for state_name in polygones.NAME:
        if state_name not in invalid_states:
            valid_states = valid_states + state_name + '|'
    return valid_states
def generate_map(init_loc,usa_polygones,data):
    election_map = folium.Map(zoom_start=2, location=init_loc, tiles='cartodbpositron')
    Choropleth(geo_data=usa_polygones[['NAME', 'geometry']].set_index('NAME').__geo_interface__,
               data=data,
               legend_name='2020 US election Donald Trump (yellow) vs Joe Biden (purple)',
               key_on="feature.id",
               fill_color='YlGnBu',
               ).add_to(election_map)
    election_map.save('E:\\study\\UNIVERSITY\\Year4 Sem A\\CS 4480\\Group2\\election_map.html')


usa_polygones = geopandas.read_file('./usapolygones/tl_2014_us_state.shx', SHAPE_RESTORE_SHX='YES')
init_long_lat = [34.19, -82.0579]
valid_states = extract_valid_states(usa_polygones)
analyzer = SentimentIntensityAnalyzer()

trump = pd.read_csv('./donaldtrump_clean.csv')
biden = pd.read_csv('./joebiden_clean.csv')
trump_biden = pd.concat([trump, biden])

tsc_df = trump_biden[['tweet', 'state', 'candidate']]
tsc_df = tsc_df.dropna()
tsc_df = tsc_df.loc[tsc_df['state'].str.contains(valid_states, case=False)]

tsc_df['sentiment'] = tsc_df.tweet.apply(analyze_sentiment)

tsc_groubBy_df = tsc_df.groupby(['state','candidate'])
tsc_groubBy_df = tsc_groubBy_df.agg({'sentiment':panda_aggregate})

usa_states =  tsc_df['state'].unique()
election_prediction_df = pd.DataFrame(index=usa_states, columns=['WhoWin'])
for state in election_prediction_df.index:
    if(tsc_groubBy_df.index.isin([(state,'trump')]).any() and tsc_groubBy_df.index.isin([(state,'biden')]).any()):
        if tsc_groubBy_df.loc[[(state,'trump')]].values > tsc_groubBy_df.loc[[(state,'biden')]].values:
            election_prediction_df.loc[state,'WhoWin'] = 1 #trump
        else:
            election_prediction_df.loc[state,'WhoWin'] = 2 #biden

generate_map(init_long_lat,usa_polygones,election_prediction_df['WhoWin'])

