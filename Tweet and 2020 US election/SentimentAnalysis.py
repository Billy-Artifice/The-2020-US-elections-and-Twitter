import numpy as np
import pandas as pd
import plotly.graph_objects as plotly_object
import nltk
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_top10_countries_tweets_count(tweet_data):
    trump_count = tweet_data.query('(candidate == "trump") ').dropna(subset=['country']).groupby(
        by='country').count().tweet.sort_values(ascending=False).head(10)
    trump_country = tweet_data.query('(candidate == "trump")').dropna(subset=['country']).groupby(
        by='country').count().tweet.sort_values(ascending=False).head(10).index
    biden_count = tweet_data.query('(candidate == "biden") ').dropna(subset=['country']).groupby(
        by='country').count().tweet.sort_values(ascending=False).head(10)
    biden_country = tweet_data.query('(candidate == "biden") ').dropna(subset=['country']).groupby(
        by='country').count().tweet.sort_values(ascending=False).head(10).index

    figure = plotly_object.Figure([plotly_object.Bar(x=trump_country, y=trump_count, name='Donald Trump'),
                                plotly_object.Bar(x=biden_country, y=biden_count, name='Joe Biden'), ])

    figure.update_layout(title_text='tweets count for top 10 countries')
    figure.update_xaxes(title='Countries')
    figure.update_yaxes(title='Tweets counts')
    figure.show()

def get_top10_states_tweets_count(tweet_data):
    trump_count = tweet_data.query('(country == "United States") & (candidate == "trump")').dropna(
        subset=['country','state']).groupby(by='state').count().tweet.sort_values(ascending=False).head(10)
    trump_state = tweet_data.query('(country == "United States") & (candidate == "trump")').dropna(
        subset=['country','state']).groupby(by='state').count().tweet.sort_values(ascending=False).head(10).index
    biden_count = tweet_data.query('(country == "United States") & (candidate == "biden")').dropna(
        subset=['country','state']).groupby(by='state').count().tweet.sort_values(ascending=False).head(10)
    biden_state = tweet_data.query('(country == "United States") & (candidate == "biden")').dropna(
        subset=['country','state']).groupby(by='state').count().tweet.sort_values(ascending=False).index

    figure = plotly_object.Figure([plotly_object.Bar(x=trump_state, y=trump_count, name='Donald Trump'),
                                plotly_object.Bar(x=biden_state, y=biden_count, name='Joe Biden'), ])

    figure.update_layout(title_text='tweets count for top 10 states')
    figure.update_xaxes(title='States')
    figure.update_yaxes(title='Tweets counts')
    figure.show()


def get_most_related_language(text):
    #{language:common_elements}
    related_languages = {}
    words = wordpunct_tokenize(text)
    words_set = set(words)
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        common_words = words_set.intersection(stopwords_set)
        related_languages[language] = len(common_words)

    most_related_language = max(related_languages, key=related_languages.get)

    return most_related_language


def analyze_sentiment(tweet_data, title):
    analyzer = SentimentIntensityAnalyzer()

    sentiments = []
    pos_sum = 0
    neg_sum = 0
    neu_sum = 0
    tweets = tweet_data.sample(frac = 1).drop_duplicates(['user_id'])[
        ['tweet', 'lat', 'long', 'country', 'state']]

    languages = [get_most_related_language(text) for text in tweets.tweet]

    usa_tweet_text = tweets.loc[np.array(languages) == 'english']
    sentences = usa_tweet_text.tweet.str.replace('\n', ' ')
    for sentence in sentences:
        sentiment_strength = analyzer.polarity_scores(sentence)
        sentiments.append(sentiment_strength['compound'])
        sentiment_value = sentiment_strength['compound']
        if (sentiment_value > 0.05):
            pos_sum = pos_sum + 1
        elif (sentiment_value < -0.05):
            neg_sum = neg_sum + 1
        else:
            neu_sum = neu_sum + 1

    colors = ['red', 'yellow','green']
    figure_labels = ["negativity","neutrality","positivity"]

    fig = plotly_object.Figure(data=[plotly_object.Funnelarea(labels=figure_labels, values=[neg_sum, neu_sum,pos_sum]) ])
    fig.update_traces(marker=dict(colors=colors))
    fig.update(layout_title_text=f'Sentiment Analysis on {title} tweets')

    fig.show()


trump = pd.read_csv('./donaldtrump_clean.csv',nrows=100000)
biden = pd.read_csv('./joebiden_clean.csv',nrows=100000)
tweet_data = pd.concat([trump, biden])
# get_top10_countries_tweets_count(tweet_data)
# get_top10_states_tweets_count((tweet_data))
analyze_sentiment(trump, 'Donald Trump')
analyze_sentiment(biden, 'Joe Biden')
