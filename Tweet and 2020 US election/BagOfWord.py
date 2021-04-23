import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import FreqDistVisualizer
from yellowbrick.datasets import load_hobbies
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def term_frequency(tweet_data, savename):
    vectorizer = CountVectorizer(max_features=500)
    docs = vectorizer.fit_transform(tweet_data['tweet'])
    feature_names = vectorizer.get_feature_names()
    visualizer = FreqDistVisualizer(features=feature_names, orient='v',)
    visualizer.fit(docs.toarray())
    save_file_pbg = savename + '.png'
    visualizer.show(save_file_pbg)
    bow_pd = pd.DataFrame(docs.toarray(), columns=feature_names)
    bow_pd.to_csv(savename + '.csv')
    return docs, vectorizer

def tfidf(tweet_data,savename):
    vectorizer = TfidfVectorizer(smooth_idf=True,max_features=500,use_idf=True,stop_words='english')
    tfidf = vectorizer.fit_transform(tweet_data['tweet'])
    feature_names = vectorizer.get_feature_names()
    dense_tfidf = tfidf.todense()
    tfidf_list = dense_tfidf.tolist()
    df = pd.DataFrame(tfidf_list, columns=feature_names)
    df.to_csv(savename+'csv')
    cloud = WordCloud(width=1600, height=800, max_font_size=200).generate_from_frequencies(df.T.sum(axis=1))
    plt.figure(figsize=(14, 12))
    plt.title(savename)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(savename +'.png')



def generateWordCloud(df,savename):
    cloud = WordCloud(width=1600, height=800, max_font_size=200).generate(' '.join(df['tweet']))
    plt.figure(figsize=(14, 12))
    plt.title(savename)
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(savename+'.png')



trump_df = pd.read_csv('./donaldtrump_clean.csv',nrows=2000)
biden_df = pd.read_csv('./joebiden_clean.csv',nrows=2000)
generateWordCloud(trump_df,'trump_tf_wordCloud')
generateWordCloud(biden_df,'biden_tf_wordCloud')
biden_tf, biden_vectorizer = term_frequency(trump_df, 'biden_tf')
trump_tf, trump_vectorizer = term_frequency(biden_df, 'trump_tf')
tfidf(trump_df,'trump_tfidf')
tfidf(biden_df,'biden_tfidf')
